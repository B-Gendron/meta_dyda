        # --------------------------------------------------- #
        # --------------------- Imports --------------------- #
        # --------------------------------------------------- #

# Torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# From others files of this repo
from preprocessing import get_pretrained_vectors, get_args_and_dataloaders

# Instantiate the Siamese Networks (MLP with ReLU or softmax)
class SiameseNetwork(nn.Module):
    '''
    A siamese network model for multi-class classification on text data.
    The two models are the same, MLP with 2 hidden layers and a last linear layer for classification.
    '''
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(SiameseNetwork, self).__init__()

        # embedding layer
        self.ebd = torch.nn.Embedding.from_pretrained(get_pretrained_vectors(), freeze=True)

        # two linear layers
        self.layer1 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)
        self.layer2 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)

        # classification layer
        self.classification_layer = torch.nn.Linear(in_features=hidden_dim, out_features=n_classes)

# Training and inference functions

# Actually train the model

# Evaluate the model