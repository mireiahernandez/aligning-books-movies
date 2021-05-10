
import ipdb
import json
import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import os.path
from collections import defaultdict
import ipdb
from S3D import get_video_feats, get_text_feats
import h5py
import argparse
import tqdm
from moviepy.editor import VideoFileClip


def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name

def get_key(i):
    key  =str(i)
    return '0'*(4 - len(key)) + key

'''
    PATHS AND NAMES
'''

dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
clips_path = '/data/vision/torralba/movies-books/booksandmovies/clips/'


movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']

movie_to_title = {movie:title for movie, title in zip(movies, movies_titles)}
movie_to_title['Shawshank.Redemption'] = 'The_Shawshank_Redemption' # fix a bug in a movie title
movie_to_imbd = {movie:imbd for movie, imbd in zip(movies, imbds)}


'''
   GET VIDEO FEATS
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    args = parser.parse_args()
    movie = args.movie
    
    # Get book sentences
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    book_sentences = book_file['book']['sentences'].item()['sentence']
    book_sentences = [sent[0][0] for sent in book_sentences]
    
    # Get clip files
    files_names = sorted(glob.glob(f"{clips_path}/{movie}/*.mp4"))

    # Compute embeddings
    image_features = get_video_feats(files_names)
    image_features = np.array(image_features)
    np.save(f"data/{args.movie}/image_features.npy", image_features)
    
    # Compute text embeddings
    text_features = get_text_feats(book_sentences)
    text_features = np.array(text_features)
    np.save(f"data/{args.movie}/text_features.npy", text_features)

