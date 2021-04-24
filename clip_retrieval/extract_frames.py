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

frames_by_number_path = '/data/vision/torralba/datasets/movies/data/frames_by_number'

text_annotation_path = '/data/vision/torralba/movies-books/booksandmovies/joanna/bksnmovies/data/gt_alignment/consecutive_text_labels_v2'
cootmil_path = '/data/vision/torralba/scratch/mireiahe/COOT-MIL/'
coot_path = '/data/vision/torralba/scratch/mireiahe/coot-videotext/'
dialog_anno = json.load(open(cootmil_path + 'data/bksnmovies/dialog_anno.json', 'r'))
#labeled_sents = json.load(open('labeled_sents.json', 'r'))
#raw_labeled_sents = json.load(open('raw_labeled_sents.json', 'r'))
movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = {movie:movie.replace('.', '_') for movie in movies}
movies_titles['Shawshank.Redemption']='The_Shawshank_Redemption'
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']
dataset_split = 1  # 1 for 90% data from all movies in train and 10% in val; 2 for n-1 movies in train 1 in val
val_movie = movies[-1]


def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name

def extract_frames(movie):
    # Get title
    title = movies_titles[movie]

    # Load shot info
    srt = scipy.io.loadmat('{}srts/{}.mat'.format(bksnmvs_path, movie))   
    
    # Load book
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    num_sents = len(book_file['book']['sentences'].item()['sentence'])

    # Get video
    print('{}/movies/{}/*.mp4'.format(dataset_path, title))
    vid_movie = glob.glob('{}/movies/{}/*.mp4'.format(dataset_path, title))
    if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.m4v'.format(dataset_path, title))
    if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.avi'.format(dataset_path, title))
    vid_movie = vid_movie[0]
    print('vid_movie is: {}'.format(vid_movie))
    
    # Define output path
    output_path = '/data/vision/torralba/movies-books/booksandmovies/frames/'+movie
    cmd = f"ffmpeg -nostats -loglevel 0 -i {vid_movie} -r 2 {output_path}/%05d.jpg"
    os.system(cmd)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    args = parser.parse_args()
    movies = ['Shawshank.Redemption','The.Green.Mile','The.Road']
    for movie in movies:
        extract_frames(movie)
