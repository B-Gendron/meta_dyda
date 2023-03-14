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

from model import SiameseNetwork

# load the model
model = SiameseNetwork(20, 300, 7)
model.load_state_dict(torch.load("./models/utterance_model.pt"))
model.eval()

        # ------------------------------------------------------ #
        # --------------------- Classifier --------------------- #
        # ------------------------------------------------------ #

class EmotionsClassifier(nn.Module):
    pass