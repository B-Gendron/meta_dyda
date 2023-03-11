# --------------------------------------------------- #
# --------------------- Imports --------------------- #
# --------------------------------------------------- #

# Torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import vocab

# Data loading and preprocessing
from datasets import load_dataset
import nltk
from nltk.tokenize import TweetTokenizer

from gensim.models.phrases import Phrases, Phraser

# General purposes modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import colored
from collections import Counter

# --------------------------------------------------- #
# ---------------- Vocabulary setup ----------------- #
# --------------------------------------------------- #

from torchtext.vocab import GloVe, vocab, FastText

pretrained_vectors = FastText(language='en')
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

def tokenize_pad_numericalize(entry, vocab_stoi, max_length=20):
    ''' 
        Performs tokenization and padding at message level.

        @param entry (str):           the sentence to process
        @param vocab_stoi (list):     a dict mapping the words to their indexes
        @param max_length (int):      length threshold for padding (default=20)

        @return padded_dialog (list): the tokenized and padded sentence 
    '''
    text = [ vocab_stoi[token] if token in vocab_stoi else vocab_stoi['<unk>'] for token in tok.tokenize(entry.lower())]
    padded_text = None
    if len(text) < max_length:   
        padded_text = text + [ vocab_stoi['<pad>'] for i in range(len(text), max_length) ] # add ones bt the end of the text and max_length
    elif len(text) > max_length: 
       padded_text = text[:max_length]
    else:                        
       padded_text = text
    return padded_text


def tokenize_all_utterances(entry, vocab_stoi, max_length=20):
    ''' 
        Apply tokenization to all the messages of each dialog. 

        @param entries (list):      list of sentences that make up the dialog
        @param vocab_stoi (list):   a dict mapping the words to their indexes
        @param max_length (int):    length threshold for padding messages(default=20)

        @return res(dict):          the tokenized and padded utterances along with the associated labels
    '''
    dialog = entry['dialog']
    emotions = entry['emotion']
    messages, labels = [], []
    for i in range(len(dialog)):
        message = tokenize_pad_numericalize(dialog[i], vocab_stoi=vocab_stoi, max_length=max_length)
        messages.append(message)
        label = emotions[i]
        labels.append(label)
    res = {'text': messages, 'label': labels}
    return res

# Apply tokenization and padding
vocab_stoi = pretrained_vocab.get_stoi()
for split in ['train', 'validation', 'test']:
 dailydialog[split] = dailydialog[split].map(lambda e: tokenize_all_utterances(e, vocab_stoi), batched=True)

# Dataset class

