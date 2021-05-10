import os
import ipdb
import glob
import json
import nltk

def decode(i):
    try:
        j = i.decode('utf-8')
        return j
    except:
        return ''



def split_into_pars(text):
    pars, i = [], 0
    while i < len(text):
        par = ''
        line = text[i]
        while line != '\n' and i < len(text):
            par += ' ' + line.lstrip().rstrip()
            i += 1
            if i < len(text): line = text[i]
        if par != '': pars.append(par.lstrip())
        i += 1
    return pars

def split_into_chaps(pars):
    chaps, i = [], 0
    while i < len(pars):
        chap = []
        par = pars[i]
        while par[:7] != 'CHAPTER' and i < len(pars):
            chap.append(par)
            i += 1
            if i < len(pars): par = pars[i]
        chaps.append(chap)
        i += 2
    return chaps[1:]

# Define raw books paths
BOOKS_PATH = "/data/vision/torralba/frames/data_acquisition/booksmovies/data/raw_books"
book_title = "Harry.Potter.and.the.Sorcerers.Stone"
book_path = glob.glob(BOOKS_PATH + '/' + book_title + '*.txt')[0]
parsed_book_path = 'parsed_books/' + book_title +'.json'

if parsed_book_path not in glob.glob('parsed_books'):
    # Read book
    with open(book_path, 'rb') as f:
        text = f.readlines()

    # Decode lines
    text = [decode(line) for line in text]

    # Split text into paragraphs
    pars = split_into_pars(text)

    # Split paragraphs into chapters
    chaps = split_into_chaps(pars)

    # Save book
    json.dump(chaps, open(parsed_book_path, 'w'))

else: print('Book already parsed')
ipdb.set_trace()
