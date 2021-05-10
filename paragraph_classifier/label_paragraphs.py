import glob
import json
import numpy as np
import nltk
import ipdb

# Paragraph labels dictionary
labels = {0:'dialog', 1:'story', 2:'descriptionOfPlace',
          3:'descriptionOfAppearance', 4:'descriptionOfAction',
         5: 'descriptionOfObject', 6:'descriptionOfSound'}

# Function that returns True if input is 'y' end False otherwise
is_true = lambda x: True if x=='1' else False

# Function to decode paragraphs
def decode(line):
    try:
        decoded_line = line.decode('utf-8')
    except:
        decoded_line = ''
    return decoded_line

# Book paths and files
book_path = '/data/vision/torralba/frames/data_acquisition/booksmovies/data/raw_books/'
book_files = glob.glob(book_path+'/*.txt')
scratch_path = '/data/vision/torralba/scratch/mireiahe/aligning-books-movies/'
labeled_paragraphs_path = scratch_path + 'paragraph_classifier/labeled_paragraphs/'

# Input prompt to label paragraphs
prompt = "Type of paragraph? \n 0:'dialog', 1:'story', 2:'descriptionOfPlace',\n 3:'descriptionOfAppearance', 4:'descriptionOfAction', 5: 'descriptionOfObject', 6:'descriptionOfSound' \n"

# Label at least 100 paragraphs for each book
for book_file in book_files:
    with open(book_file, 'rb') as f:
        # read text from book_file
        text = f.readlines()
        text = [decode(line) for line in text]
        # split into paragraphs
        paragraphs = (''.join(text)).split('\n\n')
        paragraphs = [par.replace('\n', '') for par in paragraphs] # replace end line with space

        # get book name
        book_name = book_file.split('/')[-1].split('_')[0]
        print(f"Labeling book {book_name}")

        # get labeled paragraphs file if exists
        lab_par_file = labeled_paragraphs_path + book_name + '.json'
        if lab_par_file in glob.glob(lab_par_file):
            lab_par = json.load(open(lab_par_file, 'r'))
            num_par = len(lab_par)
        else:
            lab_par = []
            num_par = 0

        # label paragraphs until we have a total of 100 paragraphs
        while len(lab_par) < 100:
            print(f"Number of labeled paragraphs: {len(lab_par)}")
            par = [np.random.choice(paragraphs)]
            print('\n', par, '\n')
            split = is_true(input('Do you want to split the paragraph into sentences? y=1, n=0'))
            if split:
                par = nltk.sent_tokenize(par[0])
            for sent in par:
                print('\n'+sent+'\n')
                skip = is_true(input('Do you want to skip? y=1, n=0'))
                if not skip:
                    label = input(prompt)
                    lab_par.append({'text': sent, 'book': book_name, 'label':label})
            stop_labeling = is_true(input('Do you want to stop? y=1, n=0'))
            if stop_labeling:
                json.dump(lab_par, open(lab_par_file, 'w'))
                break
