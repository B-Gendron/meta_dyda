# In the file main.py, we learn a representation of the utterances which is adapted to emotion detection. Now it is time to evaluate the model by using a classifier and checking for accuracy.

        # --------------------------------------------------- #
        # --------------------- Imports --------------------- #
        # --------------------------------------------------- #

# Torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Data loading and preprocessing
import json
from datasets.dataset_dict import DatasetDict

# General purposes modules
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from termcolor import colored

        # ------------------------------------------------------ #
        # --------------------- Classifier --------------------- #
        # ------------------------------------------------------ #

class EmotionsClassifier(nn.Module):
    '''
        A simple classifier that takes as an imput the representations learnt by the siamese network and outputs a class prediction
    '''
    def __init__(self, input_dim, output_dim):
        super(EmotionsClassifier, self).__init__()

        self.classification_layer = torch.nn.Linear(in_features=input_dim, out_features=output_dim)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.classification_layer(x)

        logits = self.softmax(output)
        return logits
    
