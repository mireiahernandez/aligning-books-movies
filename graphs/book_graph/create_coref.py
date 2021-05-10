import json
from tqdm import tqdm
import sys
import glob
import scipy.io
import numpy as np
import spacy
import pdb
import ipdb
import os
#from book import *
import neuralcoref
import ipdb
from stanfordcorenlp import StanfordCoreNLP



nlp = spacy.load("en")
neuralcoref.add_to_pipe(nlp)

book_title = "Harry.Potter.and.the.Sorcerers.Stone"
parsed_book_path = 'parsed_books/' + book_title +'.json'

with open(parsed_book_path, 'r') as f:
    book = json.load(f)
options = {
        'annotators': 'tokenize,ssplit,pos,depparse',
        'pipelineLanguage':'en',
        'outputFormat':'json'}

snlp = StanfordCoreNLP('stanford-corenlp-4.2.0/')
verb_pos = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
bad_verbs = ['was', 'were', 'happen', 'happened', 'be', 'know', 'knew']
object_tags = ['obl', 'obj']
all_verbs = []



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

    mid = mini
    item = cluster_info[mid]
    start_char, end_char = item['key']
    if start_char > char_range[1] or end_char < char_range[0]:
        return -1
    return mid


for ch_id, chapter in enumerate(tqdm(book)):
    doc = nlp(' '.join(chapter))
    clusters = doc._.coref_clusters
    # Build cluster

    # We add 1 because we are including a space between paragraphs
    # except in the first paragraph
    chars_per_paragraph = [len(par)+1 if i > 0 else len(par) for i,par in enumerate(chapter)]
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
            'value':  (cluster_id, name associated to the cluster (i.e, first mention)
        for example:
            [{'key': (0, 50), 'value': (0, Mr. and Mrs. Dursley, of number four, Privet Drive)},
            {'key': (75, 79), 'value': (0, Mr. and Mrs. Dursley, of number four, Privet Drive)}]
         (in this case the first mention is actually "Mr. and Mrs. .... Drive"
         and the second mention is "they" referring to "Mr. and Mrs... Drive")
    '''
    
    clusters_doc = {}
    num_clusters = len(clusters)
    for cluster in clusters:
        mentions = cluster.mentions
        cluster_info = (cluster.i, cluster.main)
        for mention in mentions:
            paragraph_id = [it for it,lp in enumerate(chars_per_paragraph) if mention.start_char < lp]
            if len(paragraph_id) == 0:
                ipdb.set_trace()
            paragraph_id = paragraph_id[0]
            if paragraph_id > 0: offset = chars_per_paragraph[paragraph_id-1]
            else: offset = 0
            chars_span = (mention.start_char-offset, mention.end_char-offset)
            if paragraph_id not in clusters_doc:
                clusters_doc[paragraph_id] = []
            clusters_doc[paragraph_id].append({'key': chars_span, 'value': cluster_info})


    relationships = []
    relationships_org = []
    for par_id, paragraph in enumerate(chapter):
        '''
            results is a dictionary with only one key 'sentences'
            
            results['sentences'] is a list of dictionaries associated
            to the different sentences
            
            results['sentences'][i] is a dictionary with keys:
                ['index', 'basicDependencies', 'enhancedDependencies', 
                'enhancedPlusPlusDependencies', 'tokens']
                
            -> results['sentences'][i]['tokens'] is a list of all the tokens
            in the sentence with the part-of-speech (pos) tag
                example of one item in results['sentences'][i]['tokens']:
                {'index': 23, 'word': 'thank', 'originalText': 'thank', 
                'characterOffsetBegin': 103, 'characterOffsetEnd': 108, 
                'pos': 'VBP', 'before': ' ', 'after': ' '}
            
            -> results['sentences'][i]['enhancedPlusPlusDependencies']
            is a list (one item per token in the sentence) with dependencies
                example of one item in the list:
                {'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 
                'dependent': 14, 'dependentGloss': 'proud'}

        '''
        results = json.loads(snlp.annotate(paragraph, options))
        offset_token = 0
        if par_id in clusters_doc.keys(): clusters_paragraph = clusters_doc[par_id]
        else: clusters_paragraph = []

        for sentence in results['sentences']:
            pos = sentence['tokens']

            # Build a dependency index
            dependent_index = {}
            governor_index = {}

            verbs = [p for p in pos if p['pos'] in verb_pos and p['word'].lower() not in bad_verbs]
            all_verbs += [v['word'] for v in verbs]
            for v in verbs:
                dependent_index[v['index']] = []
                governor_index[v['index']] = []

            for deps in sentence['enhancedPlusPlusDependencies']:
                if deps['dependent'] not in dependent_index:
                    dependent_index[deps['dependent']] = []
                dependent_index[deps['dependent']].append(deps)
                if deps['governor'] not in governor_index:
                    governor_index[deps['governor']] = []
                governor_index[deps['governor']].append(deps)
            ipdb.set_trace()
            for verb in verbs:
                #dependents_verb = dependent_index[verb['index']]
                governor_verb = governor_index[verb['index']]
                #print("Conntections")
                subject = [gov for gov in governor_verb if gov['dep'] == 'nsubj']
                obj = [gov for gov in governor_verb if gov['dep'] in object_tags]
                
                if len(subject) > 0 and len(obj) > 0:
                    st_j = [w['word'] for w in sentence['tokens']]
                    relationship = {
                        'tokens': pos,
                        'verb': verb,
                        'subject': subject,
                        'object': obj
                    }
                    relationships.append(relationship)
                    #print(' '.join(st_j))
                    #print(verb['word'])
                    #print(subject)
                    #print(obj)
                    #print('----\n')
            offset_token += len(pos)
        
        for relationship in relationships:
            # Get correfs for subject
            tokens = relationship['tokens']
            token_words = [v['word'] for v in tokens]
            token_id = relationship['subject'][0]['dependent'] - 1
            token_chars = (tokens[token_id]['characterOffsetBegin'], tokens[token_id]['characterOffsetEnd'])
            
            cluster_id = binary_search_cluster(token_chars, clusters_paragraph)
            if cluster_id == -1:
                num_clusters += 1
                cluster_id = num_clusters
                cluster_id = (cluster_id, tokens[token_id]['word'])
            else:
                cluster_id = clusters_paragraph[cluster_id]['value']

            token_id = relationship['object'][0]['dependent'] - 1
            token_chars = (tokens[token_id]['characterOffsetBegin'], tokens[token_id]['characterOffsetEnd'])
            cluster_id_obj = -1
            if len(clusters_paragraph) > 0:
                try: cluster_id_obj = binary_search_cluster(token_chars, clusters_paragraph)
                except: ipdb.set_trace()
            if tokens[token_id]['word'] == 'Chief':
                ipdb.set_trace()
            if cluster_id_obj == -1:
                num_clusters += 1
                cluster_id_obj = num_clusters
                cluster_id_obj = (cluster_id_obj, tokens[token_id]['word'])
            else:
                cluster_id_obj = clusters_paragraph[cluster_id_obj]['value']

            relation_org = [cluster_id, cluster_id_obj, relationship['verb']]
            relationships_org.append(relation_org)
            print(cluster_id, cluster_id_obj, relationship['verb']['word'], ' '.join(token_words))

            
    ipdb.set_trace()

print(set(all_verbs))

snlp.close()


