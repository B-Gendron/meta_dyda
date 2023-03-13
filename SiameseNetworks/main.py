        # --------------------------------------------------- #
        # --------------------- Imports --------------------- #
        # --------------------------------------------------- #

# Torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Data loading and preprocessing
import json
from datasets.dataset_dict import DatasetDict

# General purposes modules
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from tqdm import tqdm

# From others files of this repo
from preprocessing import get_pretrained_vectors, get_dyda_utterances

        # --------------------------------------------------- #
        # ------------------ Dataset class ------------------ #
        # --------------------------------------------------- #
        
# import again to use the right module in the class
from torch.utils.data import Dataset

# Dataset class
class DialogEmotionDataset(Dataset):
    def __init__(self, data, args):
        self.args = args
        self.data = data
        self.indexes_by_class()

    def __len__(self):
        return 1000
    
    def indexes_by_class(self):
        '''
            Classify all the utterances of the data based on their class. 

            This function, called in __init__, builds a new dictionary where the data is sorted by keys, being the 5 possible classes. 
        '''

        # get all the labels
        all_labels = np.array(deepcopy(self.data["label"])) # deepcopy instead of clone because data format is list here

        self.grouped_indexes = {}
        for i in range(0,7):
            # retrieve all the indexes for each class
            self.grouped_indexes[i] = np.where((all_labels==i))[0]
    
    def __getitem__(self, idx):
        # choose a random class for anchor and positive
        anchor_class = random.randint(0,6)
        # choose a distinct random class for negative
        negative_class = random.randint(0,6)
        while negative_class == anchor_class:
            negative_class = random.randint(0,6)

        # pick random indexes in the grouped utterances from the selected classes
        index_anchor = random.choice(self.grouped_indexes[anchor_class])
        index_positive = random.choice(self.grouped_indexes[anchor_class])
        while index_positive == index_anchor:
            index_positive = random.choice(self.grouped_indexes[anchor_class])
        index_negative = random.choice(self.grouped_indexes[negative_class])

        # retrieve the associated entries
        anchor = np.array(self.data[int(index_anchor)]["text"])
        positive = np.array(self.data[int(index_positive)]["text"])
        negative = np.array(self.data[int(index_negative)]["text"])

        item = {
          "anchor":     anchor,
          "positive":   positive,
          "negative":   negative,
          "label":      np.array([anchor_class, anchor_class, negative_class])
        }
        return item


        # --------------------------------------------------- #
        # ------------------- Dataloaders ------------------- #
        # --------------------------------------------------- #

dyda_utterances = get_dyda_utterances()

def get_args_and_dataloaders():
    args = {'bsize': 64}
    train_loader = DataLoader(dataset=DialogEmotionDataset(dyda_utterances["train"], args=args), batch_size=args['bsize'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(dataset=DialogEmotionDataset(dyda_utterances["validation"], args=args), batch_size=args['bsize'], shuffle=True, drop_last=True)
    test_loader  = DataLoader(dataset=DialogEmotionDataset(dyda_utterances["test"], args=args), batch_size=args['bsize'], shuffle=True, drop_last=True)
    return args, train_loader, val_loader, test_loader

args, train_loader, val_loader, test_loader = get_args_and_dataloaders()

# print("")
# print("Check the dimensions of the dataloader:")
# print(next(iter(train_loader))["anchor"].shape)
# print("Expected output: torch.Size([64, 20])")


        # --------------------------------------------------- #
        # ------------------- Model class ------------------- #
        # --------------------------------------------------- #

pretrained_vectors = get_pretrained_vectors()

# Instantiate the Siamese Networks (MLP with ReLU or softmax)
class SiameseNetwork(nn.Module):
    '''
    A siamese network model for multi-class classification on text data.
    The two models are the same, MLP with 1 hidden layer and a last linear layer for classification.
    '''
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(SiameseNetwork, self).__init__()

        # embedding layer
        self.ebd = torch.nn.Embedding.from_pretrained(pretrained_vectors.vectors, freeze=False)

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
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        output = triplet_loss(A, P, N)

        return output

# TEST FORWARD PATH ON ONE ITERATION:
model = SiameseNetwork(input_dim=1, hidden_dim=300, n_classes=7)
data = next(iter(train_loader))
output = model(data["anchor"][1], data["positive"][1], data["negative"][1])
print(output)
print("Expected output: tensor(0.8057, grad_fn=<MeanBackward0>)")

        # --------------------------------------------------- #
        # ------------------ Training loop ------------------ #
        # --------------------------------------------------- #

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for it, (text1, text2, text3, labels) in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch), total=train_loader.__len__()):
        text1, text2, text3 = text1.to(device), text2.to(device), text3.to(device)
        optimizer.zero_grad()
        output = model(text1, text2, text3).squeeze()
        output.backward()
        optimizer.step()

        # --------------------------------------------------- #
        # ----------------- Inference loop ------------------ #
        # --------------------------------------------------- #

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.TripletMarginLoss()

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