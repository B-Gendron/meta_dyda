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
    The two models are the same, MLP with 1 hidden layer and a last linear layer for classification.
    '''
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(SiameseNetwork, self).__init__()

        # embedding layer
        self.ebd = torch.nn.Embedding.from_pretrained(get_pretrained_vectors(), freeze=True)

        # two linear layers
        self.hidden_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)

        self.drop = nn.Dropout(p=0.25)

        # classification layer
        self.classification_layer = torch.nn.Linear(in_features=hidden_dim, out_features=n_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward_once(self, x):
        x = self.ebd(x)

        output = self.hidden_layer(x)
        output = self.drop(output)

        return output
    
    def forward(self, input1, input2, input3):
        A = self.forward_once(input1)
        P = self.forward_once(input2)
        N = self.forward_once(input3)
        output = nn.TripletMarginLoss(A, P, N)

        return output


# Training and inference functions
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for it, (text1, text2, text3, labels) in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch), total=train_loader.__len__()):
        text1, text2, text3 = text1.to(device), text2.to(device), text3.to(device)
        optimizer.zero_grad()
        output = model(text1, text2, text3).squeeze()
        output.backward()
        optimizer.step()



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
    # using default settings. After completing the 10th epoch, the average
    # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Actually train the model

# Evaluate the model