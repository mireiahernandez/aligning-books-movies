import ipdb
import json
import os
import glob
import numpy as np
import pandas as pd
import scipy.io
from collections import defaultdict
import ipdb
from itertools import chain
import argparse
import math
import torch as th

def is_recalled(probs, idx, idy, recall):
    recalled = False
    sorted_idy = np.argsort(probs[idx,:])[::-1]
    for i in range(recall):
        if abs(sorted_idy[i] - idy) < 10: recalled = True
    return recalled

def compute_is_recall(probs, idx, idy):
    r1,r5,r10 = False, False, False
    for surr_idx in range(idx - 10, idx + 10):
        if is_recalled(probs, surr_idx, idy, 1): r1 = True
        if is_recalled(probs, surr_idx, idy, 5): r5 = True
        if is_recalled(probs, surr_idx, idy, 10): r10 = True
    return r1,r5,r10

def compute_recall(probs, gt_mapping, direction):
    r1,r5,r10 = 0,0,0
    num_visual = 0
    for match in gt_mapping:
        if match['type'][1] == 1:
            num_visual += 1
            if direction == 'i2t':
                idx, idy = match['movie_ind'], match['book_ind']
            else:
                idy, idx = match['movie_ind'], match['book_ind']
            is_r1,is_r5,is_r10 = compute_is_recall(probs, idx, idy)
            if is_r1: r1 += 1
            if is_r5: r5 += 1
            if is_r10: r10 +=1 
            
    return r1/num_visual, r5/num_visual, r10/num_visual

def compute_recall_gt(probs):
    r1,r5,r10 = 0,0,0
    num  = probs.shape[0]
    for idx in range(num):
        sorted_idy = np.argsort(probs[idx,:])[::-1]
        if sorted_idy[0] == idx: r1 += 1
        for i in range(5):
            if sorted_idy[i] == idx: r5 += 1
        for i in range(10):
            if sorted_idy[i] == idx: r10 += 1

    return r1/num, r5/num, r10/num

def is_visual(lab):
    for l in lab:
        if 'descriptionOf' in l: return True

def get_visual_sentences(text_feats, labels):
    vis_idxs = []
    counter = 0
    for i, lab in labels.items():
        if is_visual(lab):
            vis_idxs.append(int(i))
            counter += 1
    return text_feats[vis_idxs, :], vis_idxs

def get_closest_visual(book_ind, visual_idxs):
    dist = 1000000
    ind = 0
    for i, idx in enumerate(visual_idxs):
        if abs(idx - book_ind) < dist:
            ind = i
            dist =  abs(idx - book_ind)
    return ind
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    parser.add_argument('--recall_over', type=str, help='gt or all')

    args = parser.parse_args()
    movie = args.movie
    
    movies=['No.Country.for.Old.Men', 'Harry.Potter.and.the.Sorcerers.Stone', 'Shawshank.Redemption', 'The.Green.Mile', 'American.Psycho', 'One.Flew.Over.the.Cuckoo.Nest', 'Brokeback.Mountain', 'The.Road']
    label_path = '/data/vision/torralba/scratch/mireiahe/aligning-books-movies/dialog_alignment/'
    
    results = {}
    

    for movie in movies:
        # Initialize results dict
        results[movie] = {}
        
        # Get labels
        labels = json.load(open(label_path + 'labeled_sents.json', 'r'))

        # Get features
        image_feats = th.FloatTensor(np.load(f"data/{movie}/image_features.npy"))
        text_feats = th.FloatTensor(np.load(f"data/{movie}/text_features.npy"))
        
        # Get only visual sentences
        visual_text_feats, visual_idxs = get_visual_sentences(text_feats, labels[movie])
        
        # Get gt mapping
        gt_mapping = json.load(open(f"data/{movie}/gt_mapping.json", 'r'))
        
        # Convert to visual indices
        gt_mapping_filtered = []
        for mapping in gt_mapping:
            if mapping['type'][1] == 1:
                movie_ind, book_ind = mapping['movie_ind'], mapping['book_ind']
                # Search for the closest visual sentence around book_ind
                book_visual_ind = get_closest_visual(book_ind, visual_idxs)
                gt_mapping_filtered.append({'movie_ind': movie_ind, 'book_ind':book_visual_ind, 'type':mapping['type']})
        
        ################### First compute normally ##################
        with th.no_grad():
            # Get similarity matrix
            logits_per_image = image_feats @ visual_text_feats.t()
            logits_per_text = visual_text_feats @ image_feats.t()
            
        # shape (Nm, Nb)
        # probs[0,:] are the probabilities that image 0 goes with each sentence in the book
        
        #probs_per_image = logits_per_image.softmax(dim=-1).cpu().numpy()
        #probs_per_text = logits_per_text.softmax(dim=-1).cpu().numpy()
        
        probs_per_image = logits_per_image.cpu().numpy()
        probs_per_text = logits_per_text.cpu().numpy()
        
        i2t = compute_recall(probs_per_image, gt_mapping_filtered, 'i2t')
        t2i = compute_recall(probs_per_text, gt_mapping_filtered, 't2i')
        
        print(f"Movie {movie}, CLIP features recall")
        print(f"recall over book i2t: r1 {i2t[0]}, r5 {i2t[1]}, r10 {i2t[2]}")
        print(f"recall over movie t2i: r1 {t2i[0]}, r5 {t2i[1]}, r10 {t2i[2]}")
        
        results[movie]['i2t'] =  i2t
        results[movie]['t2i'] = t2i
        results[movie]['visual_fraction'] = len(visual_idxs) / text_feats.shape[0]
        '''
        
        ################## If compute recall only over gt #########
        visual_mapping = [mapping for mapping in gt_mapping if mapping['type'][1] == 1]
        num_vis = len(visual_mapping)
        window_image, window_text = 10, 10
        gt_image_feats = th.zeros(size=(2*window_image*num_vis, 512)).to(device)
        gt_text_feats =  th.zeros(size=(2*window_text*num_vis, 512)).to(device)
        visual_mapping = [mapping for mapping in gt_mapping if mapping['type'][1] == 1]
        annot_mapping = []
        for i, mapping in enumerate(visual_mapping):
            book_ind, movie_ind = mapping['book_ind'], mapping['movie_ind']
            gt_image_feats[2*window_image*i:2*window_image*(i+1)] = image_feats[movie_ind-window_image:movie_ind+window_image]
            gt_text_feats[2*window_text*i:2*window_text*(i+1)] = text_feats[book_ind-window_text:book_ind+window_text]
            annot_mapping.append({
                'movie_ind': 2*window_image*i + window_image,
                'book_ind': 2*window_text*i + window_text,
                'type':mapping['type']})
        with th.no_grad():
            # Get similarity matrix
            logits_per_image = logit_scale * gt_image_feats @ gt_text_feats.t()
            logits_per_text = logit_scale * gt_text_feats @ gt_image_feats.t()
        
        #probs_per_image = logits_per_image.softmax(dim=-1).cpu().numpy()
        #probs_per_text = logits_per_text.softmax(dim=-1).cpu().numpy()
        
        probs_per_image = logits_per_image.cpu().numpy()
        probs_per_text = logits_per_text.cpu().numpy()
        
        i2t = compute_recall(probs_per_image, annot_mapping, 'i2t')
        t2i = compute_recall(probs_per_text, annot_mapping, 't2i')
        print(f"Movie {args.movie}, CLIP features recall")
        print(f"recall over gt i2t: r1 {i2t[0]}, r5 {i2t[1]}, r10 {i2t[2]}")
        print(f"recall over gt t2i: r1 {t2i[0]}, r5 {t2i[1]}, r10 {t2i[2]}")

        results[movie]['i2t_gt'] =  i2t
        results[movie]['t2i_gt'] = t2i
        '''
    json.dump(results, open('results_filtered.json', 'w'))
