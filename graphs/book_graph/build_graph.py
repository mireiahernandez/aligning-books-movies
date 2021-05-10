# Common imports
import json
from tqdm import tqdm
import sys
import glob
import scipy.io
import numpy as np
import pdb
import ipdb
import os
import nltk
import argparse
import re
from collections import defaultdict

# Import neural coreference from Hugging Face
import spacy
import neuralcoref

# Import StanfordCoreNLP model for POS and dependencies
from stanfordcorenlp import StanfordCoreNLP

# Load english neural coref model
print('Loading neuralcoref...')
nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp, max_dist_match=50, blacklist=True)
options = {
        'annotators': 'tokenize,ssplit,pos,depparse, ner',
        'pipelineLanguage':'en',
        'outputFormat':'json'}
print('neuralcoref loaded.')


# Load Stanford model (for pos and ner)
print('Loading Stanford parser...')
snlp = StanfordCoreNLP('stanford-corenlp-4.2.0/')
print('Stanford parser loaded.')

# Load NER annotations
ner = json.load(open('ner.json', 'r'))

# tags corresponding to verb pos
verb_pos = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# Load bad verbs
bad_verbs = json.load(open('bad_verbs.json', 'r'))

# Takes the index of the pos and returns
#   'typ': O/C
#   'actual_node': node if it already exists or -1 if it doesn't exist yet
#   (which means it is not a cast member nor appears in a cluster)
def get_node(index, offset, pos, cluster_par, ner_par, char_to_node):
    p  = pos[index - 1]
    span = (p['characterOffsetBegin']+offset, p['characterOffsetEnd']+offset)
    label = p['word']

    # Try to find a cluster in that span
    mention_id = search_cluster(span, cluster_par)
    
    # If a mention is found
    if mention_id != -1:
        # if there is a coref, cast will be the cast of the main mention
        actual_node = cluster_par[mention_id]['node']
        cast = cluster_par[mention_id]['cast']
        if len(cast)==0: typ='O'
        else: typ='C'
        
    # If no mention is found
    else:
        cast = get_cast(ner_par, span) 
        if cast != []:
            actual_node = [char_to_node[char] for char in cast]
            typ = 'C'
        else:
            actual_node = -1
            typ = 'O'
    
    return typ, label, actual_node

# Search for a character range in the cluster
def search_cluster(span, cluster_par):
    if cluster_par == -1: return -1
    for mention_id, item in enumerate(cluster_par):
        start, end = item['key']
        if (start, end) == span: return mention_id
    return -1

# Locates mention in chapter by identifying paragraph id
# and character span
def locate_mention_in_chapter(mention, chars_per_paragraph):
    paragraph_id = [it for it,lp in enumerate(chars_per_paragraph) if mention.start_char < lp]
    paragraph_id = paragraph_id[0]
    if paragraph_id > 0: offset = chars_per_paragraph[paragraph_id-1]
    else: offset = 0

    ### BUG ALERT: when the mention is ' Mr and Mrs..' with the space
    ### then chars_span[0] = -1
    
    
    # Get the chars span by subtracting the offset 
    # Substract 1 to account for the space at the beginning of the paragraph
    chars_span = (mention.start_char-offset-1, mention.end_char-offset-1)
    
    return paragraph_id, chars_span

def add_dialog_offset(chars_span, narratives):
    start, end = chars_span[0], chars_span[1]
    
    # Get char offset at the end of each narrative for each paragraph
    chars_per_narrative = list(np.cumsum([len(narr[2]) for narr in narratives]))

    # locate the narrative id
    narrative_id = [it for it,lp in enumerate(chars_per_narrative) if start<lp]
    narrative_id = narrative_id[0]
    # add the offset introduced by the dialogs
    if narrative_id > 0: prev_narr_len = chars_per_narrative[narrative_id-1]
    else: prev_narr_len = 0
    
    dialog_len = narratives[narrative_id][1] - prev_narr_len

    return (start + dialog_len, end + dialog_len)

# Check if there is a cast name in the span chars_span
def get_cast(ner_annotation, chars_span):
    cast = []
    for anno in ner_annotation:
        if anno[1][0] >= chars_span[0] and anno[1][1] <= chars_span[1]:
            cast.append(anno[0][0])
    return cast

# Returns clusters_doc (dictionary par_id -> mentions) and
# nodes and edges
def get_coref(chapter, chap_id):
    print('Obtaining coreference cluster...')
    
    # Split narrative and dialogs

    split_chapter = split_chap(chapter)

    text= ''
    for par in split_chapter:
        text += ' ' # Add space at the beginning
        for narr in par['narratives']: text += narr[2]

    # Obtain coref clusters with NeuralCoref
    doc = nlp(text) 
    clusters = doc._.coref_clusters
    
    # Obtain the named entity annotation dictionary for the chapter
    ner_chap = ner[str(chap_id)]
    
    # Obtain the character nodes
    char_to_node = {}
    nodes = {}
    node_index = 0
    for par_id in ner_chap.keys():
        ner_par = ner_chap[par_id]
        for ner_anno in ner_par:
            char = ner_anno[0][0]
            if char not in list(char_to_node.keys()):
                char_to_node[char] = node_index
                nodes[node_index] = {'label':char, 'typ':'C'}
                node_index += 1

    # Obtain clusters_doc
    clusters_doc = defaultdict(list)
    num_clusters = len(clusters)
    '''
        clusters_doc is a dictionary that associates paragraph indexs
        with cluster
        
        clusters_doc[par_index] is list of dictionaries corresponding
        to mentions in the paragraph
        
        clusters_doc[par_index][i] is a dictionary with two keys:
            'key': character span of the mention (relative to paragraph
                    start)
            'node': cluster_node is the index of the nodes corresponding to this mention
            'cast': cast is a list of cast names in that mention (cast = [] if there is not cast)
    '''    
    
    
    # Edges is dictionary from par_id to a list of edges for that paragraph_id
    ### Each edge is a tuple (source_node index, target_node index)
    edges = defaultdict(list)

    # Get char offset at the end of each paragraph
    chars_per_paragraph = [sum([len(narr[2]) for narr in par['narratives']])+1 for i,par in enumerate(split_chapter)] # add 1 for the added space
    chars_per_paragraph = np.cumsum(chars_per_paragraph)
    chars_per_paragraph = list(chars_per_paragraph)
    

    for cluster in clusters:
        mentions = cluster.mentions
        # Locate main mention in chapter
        main_par_id, main_chars_span = locate_mention_in_chapter(cluster.main, chars_per_paragraph)
        
        # Add dialog offset
        main_chars_span = add_dialog_offset(main_chars_span, split_chapter[main_par_id]['narratives'])
        
        
        # Define a list of cast associated with that mention
        cast = [] 
        
        # Check if main mention is a character or not and get character name
        main_ner = []
        if str(main_par_id) in ner_chap.keys(): # if there is ner in that paragraph
            main_ner = ner_chap[str(main_par_id)]
            cast = get_cast(main_ner, main_chars_span)
        
        # Add the main mention to the set of nodes
        if len(cast) == 0:
            nodes[node_index] = {'label':str(cluster.main), 'typ':'O'}
            cluster_node = [node_index]
            node_index += 1
        else: cluster_node = [char_to_node[char] for char in cast]

        # Build cluster
        for mention in mentions:
            # Locate mention in the chapter
            paragraph_id, chars_span = locate_mention_in_chapter(mention, chars_per_paragraph)
            
            # Add dialog offset
            chars_span = add_dialog_offset(chars_span, split_chapter[paragraph_id]['narratives'])

            # Add to clusters_doc
            clusters_doc[paragraph_id].append({'key': chars_span, 'node': cluster_node, 'cast':cast})

    print('Finished obtaining coreference cluster.')

    return nodes, edges, char_to_node, clusters_doc, split_chapter

re_dialog = re.compile('.*".*".*')

# Inputs a paragraph and splits dialog and narrative
# Outputs (dialog, narratives)
### -> dialogs is a list (index, dialog) where the dialog is without '"'
### -> narratives is  alist (index, offset, narrative) 
######## -> offset is the real character start relative to the beginning of the paragraph
######## -> narrative is the text
def split_par(paragraph):
    dialogs, narratives = [], []
    dialog, narrative = '', [0,'']
    building = False
    index, char_offset = 0, 0
    for character in paragraph:
        if character == '"':
            if building:
                dialogs.append((index,dialog))
                dialog = ''
                narrative[0] = char_offset + 1 #to account for "
                building=False
                index += 1
            else: 
                building=True
                if narrative[1] != '':
                    narratives.append([index] + narrative)
                    narrative = [0,'']
                    index += 1
        else:
            if building: dialog += character
            else: narrative[1] += character
        char_offset += 1
    if narrative[1] != '': narratives.append([index] + narrative)

    return dialogs, narratives

def split_chap(chapter):
    split_chapter = []
    for par_id, par in enumerate(chapter):
        dialogs, narratives =  split_par(par)
        split_chapter.append({'dialogs':dialogs, 'narratives':narratives})
    return split_chapter

# Returns nodes, edges
### nodes is a list of triples (node_index, node_label, node_type)
### edges is a dictionary where for each paragraph we store a list of 
### tuples describing the edges (source_node, target_node)
def get_nodes_and_edges(chapter, chap_id, init, end):
    
    # Obtain the named entity annotation dictionary for the chapter
    ner_chap = ner[str(chap_id)]
    
    # Get corefs
    nodes, edges, char_to_node, clusters_doc, split_chapter =  get_coref(chapter, chap_id)
    # Next node index to assign
    node_index = len(nodes)
    edge_index = 0
    # Go through each paragraph in the chapter
    for par_id in range(init, end+1):

        # Only if the paragraph contains a descritpion of Action
        #if 'descriptionOfAction' in chap_labels[par_id] or 'dialog' in chap_labels[par_id]:
        if 'descriptionOfAction' in chap_labels[par_id] or 'dialog' in chap_labels[par_id]:
            # Get the paragraph
            paragraph = chapter[par_id]
            
            # Get ner annotation for that paragraph
            if str(par_id) in ner_chap.keys(): ner_par = ner_chap[str(par_id)]
            else: ner_par = []

            # Get the cluster for that paragraph
            if par_id in clusters_doc.keys(): cluster_par = clusters_doc[par_id]
            else: cluster_par = []

            # Split narratives and dialogs
            narratives = split_chapter[par_id]['narratives']
            dialogs = split_chapter[par_id]['dialogs']

            # Apply StanfordCoreNLP to obtain a dictionary with
            # dependencies and p-o-s for each sentence
            print(split_chapter[par_id])
            # Add narratives
            for narrative in narratives:
                results = json.loads(snlp.annotate(narrative[2], options))
                offset = narrative[1]
                
                # Go through each sentence in the paragraph
                for i, sentence in enumerate(results['sentences']):
                    '''
                        pos is a list of dictionaries (each dictionary corresponds
                        to a token in the sentence)
                        p = pos[i] is a dictionary with keys:
                            -> 'index': index of the token in the sentence (1, ..., #tokens)
                            -> 'word': actual word
                            -> 'characterOffsetBegin': real start position in paragraph (0, .., len(paragraph)-1)
                            -> 'characterOffsetEnd': real end position in paragraph (0, .., len(paragraph)-1)
                            -> 'pos': part-of-speech
                            -> 'ner': named-entity-recognition (= 'PERSON' for people)
                    '''
                    pos = sentence['tokens']
                    # List of token indexs that correspond to verbs (not in the bad verbs list)
                    verbs = [p['index'] for p in pos if p['pos'] in verb_pos]
                    
                    # Get dependencies for that sentence
                    dependencies = sentence['enhancedPlusPlusDependencies']

                    # Dictionary sentence index -> typ, label and actual_node
                    sent_nodes = defaultdict(dict)
                    
                    # List of pairs of sentence indexes
                    sent_edges = defaultdict(list)
                    
                    # Boolean to store whether there is a character in the sentence
                    has_cast = False
                    
                    verbs_has_cast = defaultdict(int)
                    
                    # Add verb nodes
                    for verb in verbs:
                        sent_nodes[verb][verb] = {'typ':'A', 'label':pos[verb-1]['word'], 'actual_node':-1}
                        verbs_has_cast[verb] = False
                    
                    unseen_deps = list(range(len(dependencies)))

                    # Go through dependencies
                    for j, dependency in enumerate(dependencies):
                        gov = dependency['governor']
                        if gov in verbs:
                            
                            # Get dependency
                            deptype = dependency['dep']
                            dep = dependency['dependent']
                            
                            # If it is a relevant dependency, get node
                            if deptype in sub_dep + obj_dep or obl_dep[0].match(deptype) != None:
                                typ, label, actual_node = get_node(dep, offset, pos, cluster_par, ner_par, char_to_node)
                                if typ == 'C': verbs_has_cast[gov] += 1
                                unseen_deps.remove(j) # delete it from the unseen_deps list
                                
                                # If dependency is subject-like: add edge from subject to verb
                                if deptype in sub_dep:
                                    sent_nodes[gov][dep] = {'typ':typ, 'label': label, 'actual_node':actual_node}
                                    sent_edges[gov].append((dep, gov))
            
                                # If dependency is object-like: add edge from verb to object
                                elif deptype in obj_dep:
                                    sent_nodes[gov][dep] = {'typ':typ, 'label':label, 'actual_node':actual_node}
                                    sent_edges[gov].append((gov, dep))
                                
                                # If dependency is oblique like: add preposition and edd from verb to oblique
                                elif obl_dep[0].match(deptype) and deptype[4:]!='tmod':
                                    label = deptype[4:].replace('_', ' ') + ' ' + label # add the preposition to the label
                                    sent_nodes[gov][dep] = {'typ':typ, 'label':label, 'actual_node':actual_node}
                                    sent_edges[gov].append((gov, dep))
                    
                    # Look for other relationships in the remaining deps
                    for ind in unseen_deps:
                        dependency = dependencies[ind]
                        gov = dependency['governor']
                        if gov in sent_nodes.keys():
                            
                            deptype = dependency['dep']
                            dep = dependency['dependent']
                            '''
                            # If it is an adjective
                            
                            if deptype in att_dep:
                                sent_nodes[dep] = {'typ':'AT', 'label':dependency['dependentGloss'], 'actual_node':-1}
                                sent_edges.append((gov, dep))
                                sent_edges.append((dep, gov))
                            
                            # If it is a possessive
                            if deptype in pos_dep:
                                typ, label, actual_node = get_node(dep, offset, pos, cluster_par, ner_par, char_to_node)
                                if typ == 'C':
                                    has_cast = True
                                    sent_nodes[dep] = {'typ':typ, 'label':label, 'actual_node':actual_node}
                                    sent_edges.append((gov, dep))
                            
                            # If it's a relantionship between verbs
                            '''
                            if deptype in comp_dep:
                                # change dependent verb to 'T' type
                                sent_nodes[gov][dep] = {'typ':'T', 'label':dependency['dependentGloss'], 'actual_node':-1}
                                sent_edges[gov].append((gov, dep))
                            '''
                            # If it is an auxiliary verb
                            if deptype in aux_dep:
                                # add it to verb label
                                sent_nodes[gov]['label'] = dependency['dependentGloss'] + ' ' + sent_nodes[gov]['label']
                            '''
                    index_to_node = {}
                    # If there is cast involved somehow in that graph
                    # add it to the original graph
                    for verb in verbs_has_cast.keys():
                        if (pos[verb-1]['word'] not in bad_verbs and verbs_has_cast[verb] > 0) or \
                        (pos[verb-1]['word'] in bad_verbs and verbs_has_cast[verb] > 0):
                   

                            # Add nodes
                            for index in sent_nodes[verb].keys():
                                # if there is not a node already, add it
                                node = sent_nodes[verb][index]
                                actual_node = node['actual_node']
                                if node['actual_node'] == -1: 
                                    nodes[node_index] = {'label':node['label'], 'typ':node['typ']}
                                    index_to_node[index] = [node_index]
                                    node_index += 1
                                else:index_to_node[index] = actual_node

                            # Add edges
                            for edge in sent_edges[verb]:
                                src, tgt = index_to_node[edge[0]], index_to_node[edge[1]]
                                for s in src:
                                    for t in tgt: edges[par_id].append((s,t, edge_index))

                        '''
                        ccomp or xcomp links two different graphs where dependent is further explaining the governor
                        But the news that he was playing Seeker had leaked out somehow -  news -> playing ccomp
                        '''

                    for dialog in dialogs:
                        if i == 0 and dialog[0] == narrative[0] - 1 or \
                        (i == len(results['sentences'])-1 and dialog[0] == len(dialogs)+len(narratives)-1 and dialog[0] == narrative[0] + 1):
                            text = dialog[1]
                            root_dep = dependencies[0]
                            root_verb =  root_dep['dependent']
                            nodes[node_index] = {'label':text, 'typ':'D'}
                            if verbs_has_cast[root_verb] > 0:
                                edges[par_id].append((index_to_node[root_verb][0], node_index, edge_index))
                                node_index += 1
                                        
 

                    edge_index += 1
                        

    print('Finished obtaining nodes and edges')
    return nodes, edges


if __name__ == '__main__':
    # Load book
    book_title = "Harry.Potter.and.the.Sorcerers.Stone"
    parsed_book_path = 'parsed_books/' + book_title +'-ner.json'

    print('Loading book...')
    with open(parsed_book_path, 'r') as f:
        book = json.load(f)
    print('Book loaded.')

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter', type=int, default= 10, help='chapter number')
    parser.add_argument('--init', type=int, default=0, help='initial paragraph')
    parser.add_argument('--end', type=int, default=3, help='final paragraph')

    args = parser.parse_args()

    # Select chapter
    chap_id = args.chapter - 1
    init = args.init
    end = args.end

    # Dependency to node type
    sub_dep = ['nsubj']
    obj_dep = ['obj', 'iobj']
    obl_dep = [re.compile('^obl:.*')]
    att_dep = ['amod']
    cop_dep = ['cop']
    aux_dep = ['aux']
    pos_dep = ['nmod:poss']
    comp_dep = ['ccomp', 'xcomp']

    # Load paragraph lables
    labels = json.load(open('parsed_books/'+ book_title + '-labels.json', 'r'))
    chap_labels = labels[chap_id]


    nodes, edges = get_nodes_and_edges(book[chap_id], chap_id, init, end)
    json.dump(nodes, open('tmp/nodes.json', 'w'))
    json.dump(edges, open('tmp/edges.json', 'w'))
