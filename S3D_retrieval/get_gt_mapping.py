import h5py
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


dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)

frames_path = '/data/vision/torralba/movies-books/booksandmovies/frames/'

text_annotation_path = '/data/vision/torralba/movies-books/booksandmovies/joanna/bksnmovies/data/gt_alignment/consecutive_text_labels_v2'
cootmil_path = '/data/vision/torralba/scratch/mireiahe/COOT-MIL/'
coot_path = '/data/vision/torralba/scratch/mireiahe/coot-videotext/'
dialog_anno = json.load(open(cootmil_path + 'data/bksnmovies/dialog_anno.json', 'r'))
#labeled_sents = json.load(open('labeled_sents.json', 'r'))
#raw_labeled_sents = json.load(open('raw_labeled_sents.json', 'r'))
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = {movie:movie.replace('.', '_') for movie in movies}
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1  # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[-1]


def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name

def get_duration(srt):
    # Get scene breaks
    scene_break = srt['shot']['scene_break'][0][0] # shape (#scenes, 3)
    scene_break[:,:2] -= 1 # substract one to get actual shot id
    scene_df = pd.DataFrame(data=scene_break, columns=['start_shot','end_shot','num_shots'])
    scene_df.index.name = 'scene_id'

    # Get shots
    shot_time = srt['shot']['time'][0][0] # shape (#shots, 2)
    num_shots = shot_time.shape[0]
    shot_df = pd.DataFrame(data = shot_time, columns =['start_time', 'end_time'])
    
    # Add shot duration
    shot_df['duration'] = shot_df['end_time'] - shot_df['start_time']
    
    # Add scene id for each shot
    shot_to_scene = np.empty(num_shots, dtype='int64')
    for id_shot in range(num_shots):
        starting = (scene_df['start_shot'] <= id_shot).values
        ending = (scene_df['end_shot'] >= id_shot).values
        scene_id = np.argmax(starting & ending)
        shot_to_scene[id_shot] = scene_id
    shot_df['scene_id'] = shot_to_scene
    
  
    # Add scene start time, end time and duration to scene_df
    scene_df['scene_start_time'] = shot_df[['scene_id', 'start_time']].groupby(by='scene_id').min()
    scene_df['scene_end_time'] = shot_df[['scene_id', 'end_time']].groupby(by='scene_id').max()
    scene_df['scene_duration'] = scene_df['scene_end_time'] - scene_df['scene_start_time']

    return scene_df.iloc[-1,-2]


def get_gt_mapping(movie):
    title = movies_titles[movie]
    # Load shot info
    srt = scipy.io.loadmat('{}srts/{}.mat'.format(bksnmvs_path, movie))   
    
    # Load book
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    num_sents = len(book_file['book']['sentences'].item()['sentence'])
    
    # Get annotations
    file_anno = anno_path + movie + '-anno.mat'
    anno_file = scipy.io.loadmat(file_anno)
    annotation = anno_file['anno']
    num_align = annotation.shape[1]
    
 
    # Get frame rate
    fr = 2
    
    gt_mapping = []
    for i in range(num_align):
        if movie != 'The.Road' or i != 127: # missing data
            # get type of alignment
            match_type = [0,0,0]
            if annotation['D'][0][i].item() == 1: match_type[0] = 1
            if annotation['V'][0][i].item() == 1: match_type[1] = 1
            if annotation['S'][0][i].item() == 1: match_type[2] = 1
            
            # Get frame number (= index in the CLIP image features)
            time_sec = annotation['time'][0][i].item()
            id_clip = int(time_sec // 1.6)

            # Get sentence number (= index in the CLIP text features)
            id_sentence = annotation['sentence'][0][i].item()-1
            
            # Add to gt_mapping
            gt_mapping.append({'book_ind':id_sentence, 'movie_ind':id_clip, 'type':match_type})

    return gt_mapping

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    args = parser.parse_args()
    
    movies=['No.Country.for.Old.Men', 'Harry.Potter.and.the.Sorcerers.Stone', 'Shawshank.Redemption', 'The.Green.Mile', 'American.Psycho', 'One.Flew.Over.the.Cuckoo.Nest', 'Brokeback.Mountain', 'The.Road']
    
    for movie in movies:
        gt_mapping = get_gt_mapping(movie)
        data_path = f"data/{movie}/"
        json.dump(gt_mapping, open(data_path + '/gt_mapping.json', 'w'))
