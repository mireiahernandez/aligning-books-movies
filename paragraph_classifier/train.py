import pandas as pd
import logging
import json
import os
import math
import numpy as np

# Import transformer for MultiLabelClassification model
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)

# Import function to get metrics dictionary
from metrics import get_metrics_dict

# Import function to get evaluation plots
from evaluation import get_plots

# Logging to transformers
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# List of labels
labels_list = ['dialog', 'story', 'descriptionOfPlace', 'descriptionOfAppearance', 'descriptionOfAction', 'descriptionOfObject', 'descriptionOfSound']


# Load data
scratch_path = '/data/vision/torralba/scratch/mireiahe/'
data_path = scratch_path + 'aligning-books-movies/paragraph_classifier/labeled_paragraphs/'
d1 = json.load(open(data_path + 'paragraph_classification_data_03112020.json', 'r'))
d2 = json.load(open(data_path + 'paragraph_classification_data_TheFirm.json', 'r'))
d = d1 + d2

# Prepare data, i.e, convert labels to onehot encoding
data = []
for item in d:
    text = item['text'].replace('\n', '')
    row = [text]
    labels = item['label'].split(',')
    onehot = [ int((str(i) in labels)) for i in range(0,7)]
    data.append([text, onehot]) 
data_size = len(data)

# Input type of split
print('Random train/test split: 0, Book train/test split: 1')
split = int(input('0/1?'))
   
# Get train/test split mask
mask = np.zeros(data_size, dtype=bool)
if split == 0:
    # Get random mask
    np.random.seed(42)
    eval_frac = 0.2 # 20% for evaluation
    eval_size = math.floor(data_size*eval_frac)
    mask[:eval_size] = True
    np.random.shuffle(mask)
    
else:
    # Mask data from the last book as test
    mask[-len(d2):] = True

# Apply mask
train_data = []
eval_data = []
for i in range(data_size):
    if mask[i]:
        eval_data.append(data[i])
    else:
        train_data.append(data[i])
        
# Get data frames
train_df = pd.DataFrame(train_data)
train_df.columns = ["text", "labels"]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text", "labels"]

# Define training arguments
train_batch_size = 16
eval_batch_size = 16
num_train_epoch = 3
threshold = 0.5
steps_per_epoch = math.ceil(len(train_df) / train_batch_size)
evaluate_during_training_steps = math.ceil(steps_per_epoch/10)
save_steps = 2*num_train_epoch*steps_per_epoch # 2*number_of_total_steps so no model is saved
split_name = "random-split" if split == 0 else "book-split"
output_dir = "roberta4_{}".format(split_name)
wandb_kwargs = {"name":"{}4".format(split_name)}

train_args = {
    "threshold": threshold,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": evaluate_during_training_steps,
    "evaluate_during_training_verbose": False,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": True,
    "logging_steps": 1,
    "train_batch_size": train_batch_size,
    "eval_batch_size": eval_batch_size,
    "num_train_epochs": num_train_epoch,
    "save_steps": save_steps,
    "wandb_project": "roberta",
    "wandb_kwargs": wandb_kwargs,
    "output_dir": output_dir,
    "best_model_dir":'{}/best_model'.format(output_dir)
}

# Optional model configuration
model_args = MultiLabelClassificationArgs()

# Create a MultiLabelClassificationModel
model = MultiLabelClassificationModel(
    "roberta", "roberta-base", num_labels=7, args = model_args
)

# Define evaluation metrics
eval_metrics = get_metrics_dict(labels_list)

# Train the model
model.train_model(train_df, eval_df=eval_df, args = train_args, **eval_metrics)

# Get evaluation plots
get_plots(train_data, eval_data, labels_list, output_dir)


