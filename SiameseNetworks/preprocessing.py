        # --------------------------------------------------- #
        # --------------------- Imports --------------------- #
        # --------------------------------------------------- #

# Torch utils
from torch.utils.data import Dataset, DataLoader
import torch
from torchtext.vocab import vocab

# Data loading and preprocessing
from datasets import load_dataset, Dataset
from datasets.dataset_dict import DatasetDict
from nltk.tokenize import TweetTokenizer

# General purposes modules
import numpy as np
import random
from copy import deepcopy

        # --------------------------------------------------- #
        # ---------------- Vocabulary setup ----------------- #
        # --------------------------------------------------- #

from torchtext.vocab import vocab, FastText

# use a function to be able to call it later on in the model instantiation
def get_pretrained_vectors():
    pretrained_vectors = FastText(language='en')
    return pretrained_vectors

pretrained_vectors = get_pretrained_vectors()
pretrained_vocab = vocab(pretrained_vectors.stoi)
unk_token = "<unk>"
unk_index = 0
pad_token = '<pad>'
pad_index = 1
pretrained_vocab.insert_token("<unk>",unk_index)
pretrained_vocab.insert_token("<pad>", pad_index)
#this is necessary otherwise it will throw runtime error if OOV token is queried 
pretrained_vocab.set_default_index(unk_index)

        # --------------------------------------------------- #
        # ----------------- Data handling ------------------- #
        # --------------------------------------------------- #

dailydialog = load_dataset('daily_dialog')
tok = TweetTokenizer()

def tokenize_pad_numericalize_dialog(entry, vocab_stoi, max_length=20):
    ''' 
        Performs tokenization and padding at message level.

        @param entry (str):           the sentence to process
        @param vocab_stoi (list):     a dict mapping the words to their indexes
        @param max_length (int):      length threshold for padding (default=20)

        @return padded_dialog (list): the tokenized and padded sentence 
    '''
    dialog = [ [ vocab_stoi[token] if token in vocab_stoi else vocab_stoi['<unk>'] for token in tok.tokenize(e.lower()) ] 
            for e in entry ]
    padded_dialog = list()
    for d in dialog:
        if len(d) < max_length:    padded_dialog.append( d + [ vocab_stoi['<pad>'] for i in range(len(d), max_length) ] )
        elif len(d) > max_length:  padded_dialog.append(d[:max_length])
        else:                      padded_dialog.append(d)
    return padded_dialog


def tokenize_all_dialog(entries, vocab_stoi, max_message_length=20, max_dialog_length=12):
    ''' 
        Apply tokenization to the whole dialog. 

        @param entries (list):      list of sentences that make up the dialog
        @param vocab_stoi (list):   a dict mapping the words to their indexes
        @param max_message_length (int):    length threshold for padding messages(default=20)
        @param max_dialog_length (int):     length threshold for padding dialogs (default=12)

        @return res (dict):          the tokenized and padded utterances along with the associated labels
    '''
    res_dialog, res_labels = [], []

    for entry in entries['dialog']:
        text  = tokenize_pad_numericalize_dialog(entry, vocab_stoi)
        if len(text) < max_dialog_length:    text = text + [ [vocab_stoi['<pad>']] * max_message_length for i in range(len(text), max_dialog_length)]   # pad_message * (max_dialog_length - len(text))
        elif len(text) > max_dialog_length:  text = text[-max_dialog_length:] # keeps the last n messages
        res_dialog.append(text)

    for labels in entries['emotion']:
        if len(labels) < max_dialog_length:   labels = labels + [ 0 for i in range(len(labels), max_dialog_length) ]          # pad_label * (max_dialog_length - len(labels))
        elif len(labels) > max_dialog_length: labels = labels[-max_dialog_length:]
        res_labels.append(labels)

    res = {'text': res_dialog, 'label': res_labels}
    return res

# Apply tokenization and padding
vocab_stoi = pretrained_vocab.get_stoi()
for split in ['train', 'validation', 'test']:
    dailydialog[split] = dailydialog[split].map(lambda e: tokenize_all_dialog(e, vocab_stoi), batched=True)

# print("FIRST RECORDS ALONG WITH THE ASSOCIATED EMOTIONS")
# print("")
# print("Texts")
# print(dailydialog['train']['text'][1:5])
# print("Emotion labels")
# print(dailydialog['train']['label'][1:5])

# TODO include the code from the experiments.py file here to see if it works!

# Format the useful data in another dataset to deal with text at the utterance level (changes in dimensions)
n_train = dailydialog['train'].num_rows
n_val = dailydialog['validation'].num_rows
n_test = dailydialog['test'].num_rows

def reshape_data_utterances(max_message_length=20, max_dialog_length=12):
    x_train = np.array(dailydialog['train']['text']).reshape((max_dialog_length*n_train, max_message_length))
    x_val = np.array(dailydialog['validation']['text']).reshape((max_dialog_length*n_val, max_message_length))
    x_test = np.array(dailydialog['test']['text']).reshape((max_dialog_length*n_test, max_message_length))
    y_train = np.array(dailydialog['train']['label']).reshape((-1,1))
    y_val = np.array(dailydialog['validation']['label']).reshape((-1,1))
    y_test = np.array(dailydialog['test']['label']).reshape((-1,1))
    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = reshape_data_utterances()

def get_dyda_utterances():
    d = {'train':Dataset.from_dict({'label':y_train,'text':x_train}),
     'validation':Dataset.from_dict({'label':y_val,'text':x_val}),
     'test':Dataset.from_dict({'label':y_test,'text':x_test})
     }
    return d

dyda_utterances = get_dyda_utterances()

# UNCOMMENT THIS IF YOU NEED TO SAVE THE DATA AGAIN:
# import json
# with open("data/data_utterances.json", 'w') as f:
#     json.dump(dyda_utterances, f)

# CHECK THE FINAL DATASET STRUCTURE:
# print("")
# print("Final dataset structure:")
# print(DatasetDict(dyda_utterances))
# print("")
# print("Compare with dailydialog after preprocessing:")
# print(dailydialog)