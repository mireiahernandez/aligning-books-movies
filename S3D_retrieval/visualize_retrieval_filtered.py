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
import clip


dataset_path = '/data/vision/torralba/datasets/movies/data/'
bksnmvs_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/booksandmovies/'
anno_path = '{}/antonio/annotation/'.format(bksnmvs_path)

frames_path = '/data/vision/torralba/movies-books/booksandmovies/frames/'
frames_by_number_path = '/data/vision/torralba/datasets/movies/data/frames_by_number'


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

def get_line(title, frame_file_name, prob, log, sentence):
    line = f"<td> \
        <div><p> {title} </p></div> \
        <div> <img src='clip_viz/{frame_file_name}'> </div> \
        <div><p>{prob} ({log}): {sentence[0]} </p><div></td>"
    return line

def get_visual_sentences(text_feats, labels):
    vis_idxs = []
    counter = 0
    for i, lab in labels.items():
        if is_visual(lab):
            vis_idxs.append(int(i))
            counter += 1
    return text_feats[vis_idxs, :], vis_idxs


def is_visual(lab):
    for l in lab:
        if 'descriptionOf' in l: return True


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
    args = parser.parse_args()
    movie = args.movie
    
    label_path = '/data/vision/torralba/scratch/mireiahe/aligning-books-movies/dialog_alignment/'
    labels = json.load(open(label_path + 'labeled_sents.json', 'r'))

    
    # Load book
    book_name = getBookName(movie)
    book_file = scipy.io.loadmat(book_name)
    num_sents = len(book_file['book']['sentences'].item()['sentence'])
    
    # Get clip model
    device = "cuda" if th.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    logit_scale = model.logit_scale.exp()
    
    # Get features
    image_feats = th.FloatTensor(np.load(f"data/{movie}/image_features.npy")).to(device)
    text_feats = th.FloatTensor(np.load(f"data/{movie}/text_features.npy")).to(device)
    
    # Get gt mapping
    gt_mapping = json.load(open(f"data/{movie}/gt_mapping.json", 'r'))
    
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
    

    with th.no_grad():
        # Get similarity matrix
        logits_per_image = logit_scale * image_feats @ visual_text_feats.t()
        logits_per_text = logit_scale * visual_text_feats @ image_feats.t()
    
    # shape (Nm, Nb)
    # probs[0,:] are the probabilities that image 0 goes with each sentence in the book
    probs_per_image = logits_per_image.softmax(dim=-1).cpu().numpy()
    probs_per_text = logits_per_text.softmax(dim=-1).cpu().numpy()
    
    # Image to text
    
    # Html for visualization
    lines = ["<html>",
        "<h2>Harry Potter</h2>",
        "<table>",
        "<tbody>"
    ]
    # Get frame rate
    frame_files = sorted(glob.glob(frames_path + movie + '/*.jpg'))
    fr = 2
    
    for i, match in enumerate(gt_mapping_filtered):
        if match['type'][1] == 1:
            lines.append(f"<tr><table><tbody><th>Match {i}</th>")
            book_ind = match['book_ind']
            movie_ind = match['movie_ind']
            sentence = book_file['book']['sentences'].item()['sentence'][book_ind][0]
            frame_file = frame_files[movie_ind]
            frame_file_name = frame_file.split('/')[-1]
            
            # Visualize maximum score 

            #indx = np.argmax(probs_per_image[movie_ind-10:movie_ind+10, book_ind-10: book_ind+10])
            idx = np.argmax(probs_per_image[movie_ind-10:movie_ind+10, :])

            movie_indx, book_indx = np.unravel_index([idx], probs_per_image[movie_ind-10:movie_ind+10, :].shape)

            movie_indx = movie_ind - 10 + movie_indx[0] // 20
            book_indx = book_indx[0]
            #book_indx = book_ind - 10 + indx % 20
            # find the correct sentence (labeled as visual)
            try:
                actual_book_indx = visual_idxs[book_indx]
            except:
                ipdb.set_trace()
            prob_image = probs_per_image[movie_indx, book_indx]
            log = logits_per_image[movie_indx, book_indx]
            sentencex = book_file['book']['sentences'].item()['sentence'][actual_book_indx][0]
            frame_filex = frame_files[movie_indx]
            frame_file_namex = frame_filex.split('/')[-1]
            os.system(f"cp  {frame_filex} ~/public_html/clip_viz")
            lines.append(get_line(f"GT movie {movie_indx} - book{actual_book_indx}", frame_file_namex, prob_image, log, sentencex))
            lines.append("</tbody></table></tr>")
    lines.append("</tbody></table></html>")
    with open('visualize_retrieval_filtered.html', 'w') as out:
        out.writelines(lines)

