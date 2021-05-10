import cv2
from IPython import display
import numpy as np


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


