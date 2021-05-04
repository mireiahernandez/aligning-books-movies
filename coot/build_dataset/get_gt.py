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

'''
    UTILITIES
'''

def convertTime(value):
    second = value%60
    minute = int(value/60)%60
    hour = int(value/3600)
    return '{}:{:02}:{:02}'.format(hour, minute, second)

def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name

def getSentence(annot, seen_sentences, num_sentences_per_clip=1):
    sentences = annot['Sentences']
    sentences_labels = list(chain.from_iterable([annot['text_labels']]))
    chose_captions = [(i, x) for x, (i,j) in enumerate(zip(sentences, sentences_labels))
                      if 'dialog' not in j and 'story' not in j and j != []]
    # The sentences in annotation file are in range (max(0, alignedSentence-10),min(numberOfSentencesInBook, 30)
    # the 10th sentence corresponds to the original alignment
    center = np.argmin([np.abs(_-20) for (i, _) in chose_captions])
    sentences = [i[0] for i in chose_captions[center: len(chose_captions)] if i[0] not in seen_sentences][:num_sentences_per_clip]
    seen_sentences.append(sentences[0])
    return sentences[0], seen_sentences

'''
    PATHS AND NAMES
'''

dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
text_annotation_path = '/data/vision/torralba/movies-books/booksandmovies/joanna/bksnmovies/data/gt_alignment/consecutive_text_labels'

movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1 # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[0]
'''
    GROUND-TRUTH
'''

'''
{'duration': 122.56, 
'subset': 'validation', 
'recipe_type': '113', 
'annotations': [{'segment': [16, 25], 'id': 0, 
'sentence': 'melt butter in a pan'}, 
{'segment': [31, 34], 'id': 1, 'sentence': 'place the bread in the pan'}, {'segment': [37, 41], 'id': 2, 'sentence': 'flip the slices of bread over'}, {'segment': [43, 51], 'id': 3, 'sentence': 'spread mustard on the bread'}, {'segment': [51, 57], 'id': 4, 'sentence': 'place cheese on the bread'}, {'segment': [57, 60], 'id': 5, 'sentence': 'place the bread on top of the bread'}], 'video_url': 'https://www.youtube.com/watch?v=oDsUh1es_lo'}
'''

# Ground-truth annotations
coot = defaultdict(dict)


for i, movie in enumerate(movies):
    print('Getting ground-truth for movie: ' + movie)

    title = movies_titles[i]
    imbd = imbds[i]
    sentences_annotation = json.load(open(text_annotation_path + '_{}'.format(movies[i]), 'r'))

    
    ### SCENE AND SHOTS

    # Load shot info
    srt = scipy.io.loadmat('{}srts/{}.mat'.format(bksnmvs_path, movie))
    
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
    
    # Add scene duration to scene_df
    scene_df['scene_duration'] = shot_df[['scene_id', 'duration']].groupby(by='scene_id').sum()
    
    # Add scene start time to scene_df
    scene_df['scene_start_time'] = 0
    scene_df.loc[1:, 'scene_start_time'] = scene_df['scene_duration'].cumsum()[:-1].values
    
    # Add scenes to COOT annotations
    for scene_id in scene_df.index:
        key = movie + '_' + str(scene_id)
        coot['database'][key] = {}
        coot['database'][key]['duration'] = scene_df['scene_duration'][scene_id]
        coot['database'][key]['annotations'] = []
        if dataset_split == 1:
            if scene_id <= len(scene_df)*.9:
                coot['database'][key]['subset'] = 'training'
            else:
                coot['database'][key]['subset'] = 'validation'
        elif dataset_split == 2:
            if movie == val_movie:
                coot['database'][key]['subset'] = 'validation'
            else:
                coot['database'][key]['subset'] = 'training'


    # Join the scene_df with the shot_df
    shot_df = shot_df.merge(scene_df, on='scene_id', how='left')


    # Add the relative start_time and end_time to scene start
    shot_df['rel_start_time'] = shot_df['start_time'] - shot_df['scene_start_time']
    shot_df['rel_end_time'] = shot_df['end_time'] - shot_df['scene_start_time']

    ### GT DATAFRAME
    
    # Load book
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    
    # Get annotations
    file_anno = anno_path + movie + '-anno.mat'
    anno_file = scipy.io.loadmat(file_anno)
    annotation = anno_file['anno']
    num_align = annotation.shape[1]

    seen_sentences = []

    # Get alignment pairs
    for i in range(num_align):
        if movie != 'The.Road' or i != 127: # missing data
            # Get alignment info
            DVS = [annotation['D'][0][i].item(), annotation['V'][0][i].item(), annotation['S'][0][i].item()]
            time_sec = annotation['time'][0][i].item()
            id_sentence = annotation['sentence'][0][i].item()-1
            id_paragraph = annotation['paragraph'][0][i].item()-1
            id_line = annotation['line'][0][i].item()-1
            id_srt = annotation['srt'][0][i].item()-1
            id_shot = annotation['shot'][0][i].item()-1
            sentence = book_file['book']['sentences'].item()['sentence'][id_sentence][0]
            try:
                line = book_file['book']['lines'].item()['text'][0, id_line][0]
            except:
                line = ''
            subtitle = srt['srt']['content'][0][id_srt]
            fr = srt['shot']['fr'].item()[0][0]
            tstmp = srt['shot']['time'][0][0][id_shot]
            
            # Get scene of alignment
            id_scene = shot_df['scene_id'][id_shot]
            scene_key = movie + '_' + str(id_scene)
            sentence, seen_sentences = getSentence(sentences_annotation[i], seen_sentences)
            gt_anno = {
            'segment':[shot_df['rel_start_time'][id_shot], shot_df['rel_end_time'][id_shot]],
            'id': id_shot,
            'sentence': sentence,
            'type': DVS,
            'time':time_sec - shot_df['scene_start_time'][id_shot],
            'alig_num':i}
            coot['database'][scene_key]['annotations'].append(gt_anno)
json.dump(coot, open('bknmovies_v0_split_{}.json'.format(dataset_split)))

coot2 = defaultdict(dict)
for k,v in coot['database'].items():
    if v['annotations'] != []:
        coot2['database'][k] = v
json.dump(coot2, open('bknmovies_v0_split_{}_nonempty.json'.format(dataset_split), 'w'))