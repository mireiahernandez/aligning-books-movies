import ipdb
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import json
from tqdm import tqdm
import sys
import pickle
import networkx as nx
import numpy as np
from collections import defaultdict

DO_NER = False
DO_MATCH = False
DO_COMNAME = False

if DO_NER:
    # Load model for token classification
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # List of labels
    label_list = [
        "O",       # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",   # Beginning of a person's name right after another person's name
        "I-PER",   # Person's name
        "B-ORG",   # Beginning of an organisation right after another organisation
        "I-ORG",   # Organisation
        "B-LOC",   # Beginning of a location right after another location
        "I-LOC"    # Location
    ]

    # Labels associated with people
    person_label = ['I-PER', 'B-PER']

    # Load book
    book_title = "Harry.Potter.and.the.Sorcerers.Stone"
    parsed_book_path = 'parsed_books/' + book_title +'.json'

    with open(parsed_book_path, 'r') as f:
        book = json.load(f)

    # Dictionary with names[chap_id][par_id] = list of tuples (name, span)
    names = {}

    # Get character span in the sequence for each token
    def align_tokens(tokens, sequence):
        char_ind = 0
        token_ind = 0
        seq_len = len(sequence)
        spans = []
        while char_ind < len(sequence) and token_ind < len(tokens):
            if token_ind > len(tokens)-1:
                ipdb.set_trace()
            token = tokens[token_ind].strip('##')
            length = len(token)
            if sequence[char_ind:char_ind + length] == token:
                spans.append((char_ind, char_ind + length))
                token_ind += 1
                char_ind += length
            else:
                char_ind += 1
        return spans

    name_freq = defaultdict(int)


    for chap_id, chapter in enumerate(tqdm(book)):
        chap_names = {}
        for par_id, paragraph in enumerate(tqdm(chapter)):
            names_list = []
            # Bit of a hack to get the tokens with the special tokens
            tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(paragraph)))
            inputs = tokenizer.encode(paragraph, return_tensors="pt")
            # Run the model and get predictions
            outputs = model(inputs)[0]
            predictions = torch.argmax(outputs, dim=2)[0].tolist()
            
            # Remove start token [CLS] and end token [SEP]
            num_tokens = len(tokens)
            predictions = predictions[1:num_tokens-1]
            tokens = tokens[1:num_tokens-1]
            
            # Get character spans (needed for later replacing ner)
            spans = align_tokens(tokens, paragraph)
            
            # Variable to store a name in construction
            name_in_constr = ''
            # Variable to store the starting char number of the name
            start_char = 0
            
            # Go through each token and get named-entities
            token_ind = 0
            while token_ind < len(tokens):
                token = tokens[token_ind]
                label = label_list[predictions[token_ind]]
                # if I-PER: either a name is starting or a name is being built
                if label == 'I-PER':
                    if name_in_constr == '': # if the name is starting
                        start_char = spans[token_ind][0] # update start_char
                    name_in_constr += token.strip('##') # add token to name in constr
                # elif B-PER: this means that a name is starting after another one
                elif label == 'B-PER':
                    ipdb.set_trace()
                    start_char = spans[token_ind][0] 
                    name_in_constr = token
                # if not a person token and there is a name under construction
                # the name is now complete and append it to names_list
                elif name_in_constr != '':
                    # Check that the word is really finished
                    while token[:2] == '##':
                        name_in_constr += token[2:]
                        token_ind += 1
                        token = tokens[token_ind]
                    end_char =  spans[token_ind-1][1]
                    names_list.append((name_in_constr, (start_char, end_char)))
                    name_freq[name_in_constr] +=1
                    name_in_constr = ''
                token_ind += 1
            chap_names[par_id] = names_list
        names[chap_id] = chap_names
    json.dump(names, open('names.json', 'w'))
    json.dump(name_freq, open('name_freq.json', 'w'))


# Import movie graphs
mg_path = '/data/vision/torralba/datasets/movies/movies_graph_data/mg/py3loader'
sys.path.append(mg_path)
import GraphClasses

# Open movie graphs pkl
with open(mg_path + '/2017-11-02-51-7637_py3.pkl', 'rb') as fid:
    all_mg = pickle.load(fid, encoding='latin1')
    # latin1 is important here

# all_mg is a dictionary of MovieGraph objects
# indexed by imdb unique movie identifiers
mg = all_mg['tt0241527']  # Harry Potter

# Get movie cast
cast = [character['name'] for character in mg.castlist]

ipdb.set_trace()

if DO_MATCH:
    # Read names and name_freq
    names = json.load(open('names.json', 'r'))
    name_freq = json.load(open('name_freq.json', 'r'))
    sorted_name_freq = {k: v for k, v in sorted(name_freq.items(), key=lambda item: item[1], reverse=True)}  

    # Associate each name with a cast member:
    name_to_cast = {}
    for name in sorted_name_freq.keys():
        print('Cast list:')
        for i, c in enumerate(cast): print('{}: {}'.format(i, c))
        nums = input('Cast name for \"{}\": (0-{}, \"n\" if incorrect\n)'.format(name, len(cast)-1))
        if nums != 'n': name_to_cast[name] = [cast[int(num)] for num in nums.split(',')]
        
    # Save matches
    json.dump(name_to_cast, open('name_to_cast.json', 'w'))

if DO_COMNAME:
    # Load name dictionaries
    name_to_cast = json.load(open('name_to_cast.json', 'r'))
    names = json.load(open('names.json', 'r'))

    # Map cast names with common names:
    cast_in_book = np.unique([n for name in name_to_cast.keys() for n in name_to_cast[name]])
    cast_to_comname = {}
    for char in cast_in_book:
        print('Names already in use: {}'.format([cast_to_comname[c] for c in cast_to_comname.keys()]))
        comname = input('What name do you want to give to \"{}\" ? \n'.format(char))
        cast_to_comname[char] = comname
        
    json.dump(cast_to_comname, open('cast_to_comname.json', 'w'))

# Load book
book_title = "Harry.Potter.and.the.Sorcerers.Stone"
parsed_book_path = 'parsed_books/' + book_title +'.json'

with open(parsed_book_path, 'r') as f:
    book = json.load(f)

# Load dicitonaries
name_to_cast = json.load(open('name_to_cast.json', 'r'))
names = json.load(open('names.json', 'r'))
cast_to_comname =  json.load(open('cast_to_comname.json', 'r'))

mod_book = []
ner = {}
# Replace
for chap_ind, chap in enumerate(book):
    chap_names = names[str(chap_ind)]
    mod_chap = []
    ner_chap = defaultdict(list)
    for par_ind, par in enumerate(chap):
        if str(par_ind) in chap_names.keys():
            # Get detected names for the paragraph
            par_names = chap_names[str(par_ind)]
            # char offset due to replacement of names
            offset = 0
            for item in par_names:
                name = item[0] # name that appears in the book
                if name in name_to_cast.keys(): # if the name has a matching to cast
                    # Get the cast name
                    castname = name_to_cast[name][0]
                    # Get the mapped common name
                    comname = cast_to_comname[castname]
                    # Get original start and end char
                    [start, end] = item[1]
                    # Replace actual name by common name
                    par = par[:start+offset] + comname  + par[end+offset:]
                    # Get the new actual start and end char
                    actual_start = start + offset
                    actual_end = actual_start + len(comname)
                    # Save to dict for later
                    ner_chap[par_ind].append([castname, [actual_start, actual_end]])
                    # Update the offset
                    offset += len(comname) - (end - start)
        mod_chap.append(par)
    mod_book.append(mod_chap)
    ner[chap_ind] = ner_chap

json.dump(mod_book, open('parsed_books/' + book_title + '-ner.json', 'w'))
json.dump(ner, open('ner.json', 'w'))
