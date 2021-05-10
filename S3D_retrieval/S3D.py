import os
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

import numpy as np
import math
import ipdb
import time
import tqdm

import cv2
from IPython import display
import numpy as np


# Load the model once from TF-Hub.
hub_handle = 'https://tfhub.dev/deepmind/mil-nce/s3d/1'
hub_model = hub.load(hub_handle)

# Define video loading and visualization functions  { display-mode: "form" }

# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]


def load_video(video_path, max_frames=32, resize=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    frames = np.array(frames)
    if len(frames) < max_frames:
        n_repeat = int(math.ceil(max_frames / float(len(frames))))
        frames = frames.repeat(n_repeat, axis=0)
    frames = frames[:max_frames]
    return frames / 255.0

def display_video(urls):
    html = '<table>'
    html += '<tr><th>Video 1</th><th>Video 2</th><th>Video 3</th></tr><tr>'
    for url in urls:
        html += '<td>'
        html += '<img src="{}" height="224">'.format(url)
        html += '</td>'
    html += '</tr></table>'
    return display.HTML(html)

def display_query_and_results_video(query, urls, scores):
    """Display a text query and the top result videos and scores."""
    sorted_ix = np.argsort(-scores)
    html = ''
    html += '<h2>Input query: <i>{}</i> </h2><div>'.format(query)
    html += 'Results: <div>'
    html += '<table>'
    html += '<tr><th>Rank #1, Score:{:.2f}</th>'.format(scores[sorted_ix[0]])
    html += '<th>Rank #2, Score:{:.2f}</th>'.format(scores[sorted_ix[1]])
    html += '<th>Rank #3, Score:{:.2f}</th></tr><tr>'.format(scores[sorted_ix[2]])
    for i, idx in enumerate(sorted_ix):
        url = urls[sorted_ix[i]];
        html += '<td>'
        html += '<img src="{}" height="224">'.format(url)
        html += '</td>'
    html += '</tr></table>'
    return html


# Define video and text encoders

def encode_video(model, input_frames):
    """Generate embeddings from the model from video frames """
    # Input_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
    vision_output = model.signatures['video'](tf.constant(tf.cast(input_frames, dtype=tf.float32)))
    return vision_output['video_embedding']

def encode_text(model, input_words):
    """Generate embeddings from the model from input words."""
    # Input_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
    text_output = model.signatures['text'](tf.constant(input_words))
    return text_output['text_embedding']



# Load example videos and define text queries  { display-mode: "form" }
def get_video_feats(files):
    num_videos = len(files)

    # Split into batches of batch_size = 32
    splits = [32*i for i in range(0, int(np.floor(num_videos/32))+1)]
    splits.append(num_videos)
    video_embd = []
    # Generate the embeddings for each batch
    for i in tqdm.tqdm(range(len(splits)-1)):
        # read start and end index
        start, end= splits[i], splits[i+1]
        if start != end:
            # read videos
            all_videos = []
            for path in files[start:end]:
                try:
                    all_videos.append(load_video(path))
                except:
                    print(path)
                    all_videos.append(np.zeros(shape=(32, 224, 224, 3)))
            videos_np = np.stack(all_videos, axis=0)
            # calculate embeddings
            embd = encode_video(hub_model, videos_np)
            video_embd.append(embd)

    # Concatenate batches
    video_embd = tf.concat(video_embd, axis=0)

    return video_embd


def get_text_feats(sentences):
    num_sents = len(sentences)

    # Split into batches of batch_size = 32
    splits = [32*i for i in range(0, int(np.floor(num_sents/32))+1)]
    splits.append(num_sents)
    text_embd = []
    # Generate the embeddings for each batch
    for i in tqdm.tqdm(range(len(splits)-1)):
        # read start and end index
        start, end= splits[i], splits[i+1]
        # calculate embeddings
        embd = encode_text(hub_model, sentences[start:end])
        text_embd.append(embd)

    # Concatenate batches
    text_embd = tf.concat(text_embd, axis=0)
    return text_embd
