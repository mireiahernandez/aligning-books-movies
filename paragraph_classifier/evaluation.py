import pandas as pd
import logging
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Get dataframe in correct format
def get_df(d, labels_list):
    data = []
    for item in d:
        text = item[0]
        row = [text]
        onehot = item[1]
        for index in onehot:
            row.append(index)
        data.append(row)
    df = pd.DataFrame(data)
    columns = ["text"]
    for label in labels_list:
        columns.append(label)
    df.columns =  columns
    return df

# Categories distribution
def get_parcount(df, labels_list, filename):
    categories = list(df.columns.values)[1:]
    sns.set(font_scale = 1)
    plt.figure(figsize=(15,8))
    ax= sns.barplot(categories, df.iloc[:,1:].sum())
    plt.title("Paragraphs in each category", fontsize=24)
    plt.ylabel('Number of paragraphs', fontsize=18)
    plt.xlabel('Paragraph Type ', fontsize=18)#adding the text labels
    rects = ax.patches
    labels = df.iloc[:,1:].sum().values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)
    plt.savefig(filename)

def get_labcount(df, labels_list, filename):
    rowSums = df.iloc[:,1:].sum(axis=1)
    multiLabel_counts = pd.Series.sort_index(rowSums.value_counts())
    sns.set(font_scale = 2)
    plt.figure(figsize=(15,8))
    ax = sns.barplot(multiLabel_counts.index, multiLabel_counts.values)
    plt.title("Paragraphs having multiple labels")
    plt.ylabel('Number of paragraphs', fontsize=18)
    plt.xlabel('Number of labels', fontsize=18)#adding the text labels
    rects = ax.patches
    labels = multiLabel_counts.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    plt.savefig(filename)
    


def get_plots(train_data, eval_data, labels_list, output_dir):
    train_df = get_df(train_data, labels_list)
    eval_df = get_df(eval_data, labels_list)
    
    # Categories distribution
    get_parcount(train_df, labels_list, '{}/parcount_train.png'.format(output_dir))
    get_parcount(eval_df, labels_list, '{}/parcount_eval.png'.format(output_dir))
    
    # # Number of labels
    get_labcount(train_df, labels_list, '{}/labcount_train.png'.format(output_dir))
    get_labcount(eval_df, labels_list, '{}/labcount_eval.png'.format(output_dir))

