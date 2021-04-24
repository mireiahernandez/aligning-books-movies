import argparse
import ipdb
import glob
import numpy as np
import scipy.io
from clip_model.clip_encoder import clip_image_encoder
from clip_model.clip_encoder import clip_text_encoder

def getBookName(filen):
    filen.split('/')[-1]
    book_path = '{}/books/'.format(bksnmvs_path)
    book_name = filen+'.(hl-2).mat'
    return book_path+book_name




dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)
frames_path = '/data/vision/torralba/movies-books/booksandmovies/frames/'
frames_by_number_path = '/data/vision/torralba/datasets/movies/data/frames_by_number/'


movies = ['American.Psycho','Brokeback.Mountain','Fight.Club','Gone.Girl','Harry.Potter.and.the.Sorcerers.Stone','No.Country.for.Old.Men','One.Flew.Over.the.Cuckoo.Nest','Shawshank.Redemption','The.Firm','The.Green.Mile','The.Road']
movies_titles = [movie.replace('.', '_') for movie in movies]
imbds = ['tt0144084','tt0388795','tt0137523','tt2267998','tt0241527','tt0477348','tt0073486','tt0111161','tt0106918','tt0120689','tt0898367']




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
    
    # Get frame files
    frame_files = sorted(glob.glob(frames_path + movie + '/*.jpg'))
    fr = 2
    '''
    # Get server frame files
    frame_dirs = sorted(glob.glob(frames_by_number_path + movie.replace('.','_') +'/*'))
    frame_files = []
    for frame_dir in frame_dirs:
        frame_files.extend(sorted(glob.glob(frame_dir + '/*.jpg')))
    
    duration = 9139.24
    server_fr = len(frame_files)/duration
    
    feat_frame_files =[]
    
    for i in range(len(extracted_frame_files)):
        time_movie =max(i/fr -0.5, 0)
        frame_num = min(int(round(time_movie * server_fr)), len(frame_files)-1)
        try:
            feat_frame_files.append(frame_files[frame_num])
        except:
            ipdb.set_trace()
    '''
    
    # Get book "image" through CLIP text encoder    
    image_features = clip_image_encoder(frame_files)
    image_features = np.array(image_features)
    np.save(f"data/{args.movie}/image_features.npy", image_features)
    print(f"Image features saved at: data/{args.movie}/image_features.npy")
    
    # Get movie "image" through CLIP image encoder
    text_features = clip_text_encoder(book_sentences)
    text_features = np.array(text_features)
    np.save(f"data/{args.movie}/text_features.npy", text_features)
    print(f"Text features saved at: data/{args.movie}/text_features.npy")
