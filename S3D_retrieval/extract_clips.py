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
import h5py
import argparse
import tqdm
from moviepy.editor import VideoFileClip


def get_key(i):
    key  =str(i)
    return '0'*(4 - len(key)) + key

'''
    PATHS AND NAMES
'''

dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
clip_path  = '/data/vision/torralba/movies-books/booksandmovies/clips/'
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
    movies = ['American.Psycho','Brokeback.Mountain','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Green.Mile','The.Road']

    parser = argparse.ArgumentParser()
    parser.add_argument('--movie', type=str, help='movie name')
    args = parser.parse_args()
    movie = args.movie
    
    for movie in movies:
        print('Getting video features for: ' + movie)

        title = movie_to_title[movie]
        imbd = movie_to_imbd[movie]
        
        # path to movie video
        vid_movie = glob.glob('{}/movies/{}/*.mp4'.format(dataset_path, title))
        if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.m4v'.format(dataset_path, title))
        if vid_movie == []: vid_movie = glob.glob('{}/movies/{}/*.avi'.format(dataset_path, title))
        vid_movie = vid_movie[0]
        print('vid_movie is: {}'.format(vid_movie))
        
        # get movie duration
        clip_movie = VideoFileClip(vid_movie)
        duration =  clip_movie.duration
        start_time = 0
        
        ticks = [start_time + 1.6*i for i in range(0, int(np.ceil(duration/1.6)))]
        ticks.append(start_time+duration)
        segments = [[ticks[i-1], ticks[i+1]] for i in range(1, len(ticks)-1)]
        '''
        # Remove contents of tmp 
        files = glob.glob('tmp/*mp4')
        for f in files: os.remove(f)
        print('Removed tmp contents')
        '''
        
        file_names = []
        for i, [start, end] in enumerate(tqdm.tqdm(segments)):
            key = get_key(i)
            start = np.round(start, decimals=1)
            end = np.round(end, decimals=1)
            dur = np.round(end-start, decimals=1)
            out_name = f"{clip_path}/{movie}/{movie}_{key}.mp4"
            file_names.append(out_name)
            #print('start: {}, end: {}, duration: {}'.format(start, end, dur))
            if not os.path.isfile(out_name): #only extract if not extracted yet
                cmd =  'ffmpeg -nostats -loglevel 0 -ss {} -i {} -t {} -vcodec libx264 -acodec aac -strict -2 {}'.format(start, vid_movie, dur, out_name)
                os.system(cmd)

