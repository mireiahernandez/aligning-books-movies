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
import ipdb
import nltk
from collections import defaultdict

# Import neural coreference from Hugging Face
import spacy
import neuralcoref

# Import StanfordCoreNLP model for POS and dependencies
from stanfordcorenlp import StanfordCoreNLP

# Load english neural coref model
nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp, max_dist_match=50, blacklist=True)
# Load Stanford model (for pos and ner)
options = {
        'annotators': 'tokenize,ssplit,pos,depparse, ner',
        'pipelineLanguage':'en',
        'outputFormat':'json'}

snlp = StanfordCoreNLP('stanford-corenlp-4.2.0/')

# Load NER annotations
ner = json.load(open('ner.json', 'r'))

# Load bad verbs
bad_verbs = json.load(open('bad_verbs.json', 'r'))

# tags corresponding to verb pos
verb_pos = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# tags corresponding to subject/object dependencies
sub_dep, obj_dep, aux_dep = ['nsubj'], ['obj', 'iobj'], ['aux']

'''
# Initialize a verb list
verb_list = []
'''

# Get a dictionary for a word with:
#   'span': character span relative to paragraph start
#   'word': actual word
#   'mention_id': id of the mention in the cluster_par (-1 if not in cluster)
#   'cast': name of the cast member if it is a charaacter, (-1 otherwise)
def get_pos_info(index, pos, cluster_par, ner_par):
    p  = pos[index - 1]
    span = (p['characterOffsetBegin'], p['characterOffsetEnd'])
    word = p['word']
    # For subject and object (not the verb) look for coref in cluster
    if cluster_par != -1:
        mention_id = binary_search_cluster(span, cluster_par)
        # if there is a coref, cast will be the cast of the main mention
        if mention_id != -1: cast = cluster_par[mention_id]['main'][4]
        # If there is not, search in the ner par
        else: cast = get_cast(ner_par, span) 
        return {'span': span, 'word': word, 'mention_id': mention_id, 'cast': cast}
    else: return {'span':span, 'word': word}

# Search for a character range in the cluster
def binary_search_cluster(char_range, cluster_info):
    mini, maxi = 0, len(cluster_info)
    while maxi - mini > 1:
        mid = int((maxi + mini) / 2)
        item = cluster_info[mid]
        start_char, end_char = item['key']
        if start_char > char_range[1]:
            maxi = mid
        elif end_char < char_range[0]:
            if mini == mid:
                # No match found
                return -1
            mini = mid
        else:
            return mid
    return -1

    mid = mini
    item = cluster_info[mid]
    start_char, end_char = item['key']
    if start_char > char_range[1] or end_char < char_range[0]:
        return -1
    return mid

# Locates mention in chapter by identifying paragraph id
# and character span
def locate_mention_in_chapter(mention, chars_per_paragraph):
    paragraph_id = [it for it,lp in enumerate(chars_per_paragraph) if mention.start_char < lp]
    if len(paragraph_id) == 0:
        ipdb.set_trace()
    paragraph_id = paragraph_id[0]
    if paragraph_id > 0: offset = chars_per_paragraph[paragraph_id-1]
    else: offset = 0
    # Get the chars span by subtracting the offset 
    # Substract 1 to account for the space at the beginning of the paragraph
    chars_span = (mention.start_char-offset-1, mention.end_char-offset-1)
    ### BUG ALERT: when the mention is ' Mr and Mrs..' with the space
    ### then chars_span[0] = -1
    return paragraph_id, chars_span

# Check if there is a cast name in the span chars_span
def get_cast(ner_annotation, chars_span):
    cast = -1
    for anno in ner_annotation:
        if anno[1][0] >= chars_span[0] and anno[1][1] <= chars_span[1]:
            cast = anno[0][0]
    return cast


def strip_dialogs(chapter):
    stripped_chapter = []
    for par in chapter:
        if '\"' not in par: stripped_chapter.append(par)
        else: stripped_chapter.append('')
    return stripped_chapter


# Returns nodes, edges
### nodes is a list of triples (node_index, node_label, node_type)
### edges is a dictionary where for each paragraph we store a list of 
### tuples describing the edges (source_node, target_node)
def get_nodes_and_edges(chapter, chap_id):
    ### 1) Obtain coreferences
    '''
    chapter = strip_dialogs(chapter)
    '''
    print('Obtaining coreference cluster...')
    doc = nlp(' ' + ' '.join(chapter)) # Add space at the beginning of each par
    clusters = doc._.coref_clusters
    ner_chap = ner[str(chap_id)]
    # Build cluster
    # We add 1 because we are including a space before each paragraph
    chars_per_paragraph = [len(par)+1 for i,par in enumerate(chapter)]
    chars_per_paragraph = np.cumsum(chars_per_paragraph)
    chars_per_paragraph = list(chars_per_paragraph)
    '''
        clusters_doc is a dictionary that associates paragraph indexs
        with cluster
        
        clusters_doc[par_index] is list of dictionaries corresponding
        to mentions in the paragraph
        
        clusters_doc[par_index][i] is a dictionary with two keys:
            'key': character span of the mention (relative to paragraph
                    start)
            'main': info about the main mention (cluster_index, par_id, char_span, name, cast)
            -> cluster_index uniquely idenfifies the cluster
            -> par_id uniquely indentifies the paragraph
            -> char_span is the span in characters of the main mention
            -> name is the actual text of the main mention
            -> cast is the name of the character if it is one (ow it is -1)
    '''
    print('Finished obtaining coreference cluster')
    print('Obtaining nodes and edges...')
    # Nodes is a list with all of the nodes in the chapter
    ### Each node is a triple (node_id, node_label, node_type)
    nodes = []
    
    # Edges is dictionary from par_id to a list of edges for that paragraph_id
    ### Each edge is a tuple (source_node index, target_node index)
    edges = defaultdict(list)

    clusters_doc = {}
    num_clusters = len(clusters)
    for cluster in clusters:
        mentions = cluster.mentions
        # Locate main mention in chapter
        main_par_id, main_chars_span = locate_mention_in_chapter(cluster.main, chars_per_paragraph)
        
        # Check if main mention is a character or not and get character name
        main_ner = []
        if str(main_par_id) in ner_chap.keys(): main_ner = ner_chap[str(main_par_id)]
        cast = get_cast(main_ner, main_chars_span)
        
        # Add all the information
        cluster_info = (cluster.i, main_par_id, main_chars_span, str(cluster.main), cast)
        
        # Add the main mention to the set of nodes
        if cast == -1: nodes.append((cluster.i, str(cluster.main), 'O'))
        else: nodes.append((cluster.i, cast, 'C'))

        # Build cluster
        for mention in mentions:
            # Locate mention in the chapter
            paragraph_id, chars_span = locate_mention_in_chapter(mention, chars_per_paragraph)
            if paragraph_id not in clusters_doc:
                clusters_doc[paragraph_id] = []
            clusters_doc[paragraph_id].append({'key': chars_span, 'main': cluster_info})

    ### 2) Obtain dependencies

    node_index = len(nodes) # next node index to assign
    # Go through each paragraph in the chapter
    for par_id, paragraph in enumerate(tqdm(chapter)):
        # Get ner annotation for that paragraph
        if str(par_id) in ner_chap.keys(): ner_par = ner_chap[str(par_id)]
        else: ner_par = []

        # Get the cluster for that paragraph
        if par_id in clusters_doc.keys(): cluster_par = clusters_doc[par_id]
        else: cluster_par = []
        # Apply StanfordCoreNLP to obtain a dictionary with
        # dependencies and p-o-s for each sentence
        results = json.loads(snlp.annotate(paragraph, options))

        # Go through each sentence in the paragraph
        for i, sentence in enumerate(results['sentences']):
            '''
                pos is a list of dictionaries (each dicitonary corresponds
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
            verbs = [p['index'] for p in pos if p['pos'] in verb_pos and p['word'] not in bad_verbs]

            # Get subject and object for the verbs in the sentence
            triplets = {verb_index:{'subject':[], 'object':[], 'oblique': [], 'auxiliary':[]} for verb_index in verbs}
            for deps in sentence['enhancedPlusPlusDependencies']:
                if deps['governor'] in verbs:
                    if i == 4 and par_id == 109: print(deps)
                    # Get subject
                    if deps['dep'] in sub_dep:
                        subjct = deps['dependent']
                        triplets[deps['governor']]['subject'].append(subjct)
                    # Get object
                    elif deps['dep'] in obj_dep:
                        objct = deps['dependent']
                        triplets[deps['governor']]['object'].append(objct)
                    # Get auxiliary verbs
                    elif deps['dep'] in aux_dep:
                        aux = deps['dependent']
                        triplets[deps['governor']]['auxiliary'].append(aux)
                    elif deps['dep'][:3] == 'obl'and deps['dep'][4:]!= 'tmod':
                        obl = deps['dependent']
                        full_clause= deps['dep'][4:] + ' ' + deps['dependentGloss']
                        triplets[deps['governor']]['oblique'].append([obl, full_clause])
                        
            
            # Get real character positions for complete triplets (subj+verb+obj)
            for verb_index in verbs:
                triplet = triplets[verb_index]
                sublist, objlist, obllist, auxlist = triplet['subject'], triplet['object'], triplet['oblique'], triplet['auxiliary']

                # Get dictionaries for subject, object and verb
                sub_dicts = [get_pos_info(sub, pos, cluster_par, ner_par) for sub in sublist]
                obj_dicts = [get_pos_info(obj, pos, cluster_par, ner_par) for obj in objlist]
                verb_dict = get_pos_info(verb_index, pos, -1, -1)
                aux_dicts = [get_pos_info(aux, pos, -1, -1) for aux in auxlist]
                
                # Check if interaction involves cast members
                has_cast = False
                for dic in sub_dicts + obj_dicts:
                    if dic['cast'] != -1: has_cast = True
                
                
                if has_cast:
                    # Create a node for the verb
                    verb_node = node_index
                    verb_label = verb_dict['word']
                    # Add auxiliary verbs if there are any
                    for aux_dict in aux_dicts:
                        if aux_dict['span'][1] < verb_dict['span'][0]: verb_label = aux_dict['word'] + ' ' + verb_label
                        else: verb_label += ' ' + aux_dict['word']

                    nodes.append((verb_node, verb_label, 'A'))
                    node_index += 1
                    # Add oblique clauses
                    for obl in obllist:
                        nodes.append((node_index, obl[1], 'O'))
                        # Add edge
                        edges[par_id].append((verb_node, node_index, (par_id, verb_index)))
                        node_index += 1
                    
                    # Add edges from subjects to verbs
                    for sub_dict in sub_dicts:
                        mention_id = sub_dict['mention_id']
                        # If there is a coreference
                        if mention_id != -1:
                            node_ind = cluster_par[mention_id]['main'][0]
                            node_type = nodes[node_ind][2]
                        # If there is not, create a new node
                        else:
                            # Create a new node
                            node_ind = node_index
                            # Check if it's a cast name
                            cast = sub_dict['cast']
                            if cast != -1: nodes.append((node_ind, cast, 'C'))
                            else: nodes.append((node_ind, sub_dict['word'], 'O'))
                            node_index += 1
                        edges[par_id].append((node_ind, verb_node, (par_id, verb_index)))
                    # Add edges form verbs to objects
                    for obj_dict in obj_dicts:
                        mention_id = obj_dict['mention_id']
                        # If there is a coreference
                        if mention_id != -1:
                            node_ind = cluster_par[mention_id]['main'][0]
                            node_type = nodes[node_ind][2]
                        # If there is not, create a new node
                        else:
                            # Create a new node
                            node_ind = node_index
                            # Check if it's a cast name
                            cast = obj_dict['cast']
                            if cast != -1: nodes.append((node_ind, cast, 'C'))
                            else: nodes.append((node_ind, obj_dict['word'], 'O'))
                            node_index += 1
                        edges[par_id].append((verb_node, node_ind, (par_id, verb_index)))
                '''
                # Append to paragraph interactions list
                par_inter.append({'subject':sub_dicts, 'object':obj_dicts, 'verb':verb_dict})
                
                
                # Print triplet for qualitative evaluation
                triplet = ([sub_dict['coref'] for sub_dict in sub_dicts], 
                            verb_dict['word'], 
                            [obj_dict['coref'] for obj_dict in obj_dicts])
                print(triplet)
                # print sentence for context
                print(paragraph[pos[0]['characterOffsetBegin']:pos[-1]['characterOffsetEnd']])
                '''
                '''
                # Add verb to verb list
                verb_list.append(verb_dict['word'])
                '''
    print('Finished obtaining nodes and edges')
    return nodes, edges

'''
# Load book
book_title = "Harry.Potter.and.the.Sorcerers.Stone"
parsed_book_path = 'parsed_books/' + book_title +'-ner.json'

with open(parsed_book_path, 'r') as f:
    book = json.load(f)

# Load NER annotations
ner = json.load(open('ner.json', 'r'))

# Get graph info
chapter = book[0]
nodes, edges = get_graph_info(chapter, 0)
ipdb.set_trace()


# Print verbs and filter out the most frequent non-visual verbs
verb_arr = np.array(verb_list)
(unique, counts) = np.unique(verb_arr, return_counts=True)
sorted_verbs = [unique[arg] for arg in np.argsort(counts)]
sorted_verbs.reverse()
sorted_counts = np.sort(counts).tolist()
sorted_counts.reverse()
bad_verbs = []
for i, verb in enumerate(sorted_verbs):
    if sorted_counts[i] > 5:
        is_visual = input('Is \"{}\" a visual verb? y/n \n'.format(verb))
        if is_visual == 'n':
            bad_verbs.append(verb)

json.dump(bad_verbs, open('bad_verbs.json', 'w'))
'''
