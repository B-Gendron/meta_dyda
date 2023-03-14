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
args['max_eps'] = 10

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
# model = SiameseNetwork(input_dim=20, hidden_dim=300, n_classes=7)
# data = next(iter(train_loader))
# output = model(data["anchor"][1], data["positive"][1], data["negative"][1])
# print(output)
# print("Expected output: tensor(0.8057, grad_fn=<MeanBackward0>)")

        # --------------------------------------------------- #
        # ------------------ Training loop ------------------ #
        # --------------------------------------------------- #

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_it = []

    for it, batch in tqdm(enumerate(train_loader), desc="Epoch %s: " % (epoch), total=train_loader.__len__()):

        batch = {'anchor': batch['anchor'].to(device), 'positive': batch['positive'].to(device), 'negative' : batch['negative'].to(device), 'label': batch['label'].to(device)}

        optimizer.zero_grad()

        # perform training
        output = model(batch['anchor'], batch['positive'], batch['negative'])
        output.backward()
        optimizer.step()

        # store loss history
        loss_it.append(output.item())

    # print useful information about the training progress and scores on this training set's full pass (i.e. 1 epoch)
    print("Epoch %s/%s - %s : (%s %s)" % (colored(str(epoch), 'blue'),args['max_eps'] , colored('Training', 'blue'), colored('average loss', 'cyan'), sum(loss_it)/len(loss_it)))

    # return the loss history so we can plot it later
    return loss_it

# TEST ON ONE EPOCH
# model = SiameseNetwork(input_dim=20, hidden_dim=300, n_classes=7)
# optimizer = optim.Adam(model.parameters(), lr = 1e-3)
# device = torch.device("cuda" if torch.cuda.is_available() else 'mps')
# model.to(device)
# print("Device: ", device)
# train(args=args, model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=1)

        # --------------------------------------------------- #
        # ----------------- Inference loop ------------------ #
        # --------------------------------------------------- #

def test(target, model, loader, device):
    model.eval()
    loss_it = []

    for it, batch in tqdm(enumerate(loader), desc="%s: " % (target), total=loader.__len__()):

        with torch.no_grad():
            
            batch = {'anchor': batch['anchor'].to(device), 'positive': batch['positive'].to(device), 'negative' : batch['negative'].to(device), 'label': batch['label'].to(device)}

            output = model(batch['anchor'], batch['positive'], batch['negative'])

            loss_it.append(output.item())

    loss_it_avg = sum(loss_it)/len(loss_it)

    # print useful information. Important during training as we want to know the performance over the validation set after each epoch
    print("%s : (%s %s)" % ( colored(target, 'blue'), colored('average loss', 'cyan'), sum(loss_it)/len(loss_it)))

    return loss_it_avg


        # --------------------------------------------------- #
        # ----------------- Actual training ----------------- #
        # --------------------------------------------------- #

def run_epochs(model, args, optimizer, train_loader, device):
    val_ep_losses = []

    for ep in range(args['max_eps']):
        # train the model on the train loader
        train(args, model, device, train_loader, optimizer, ep)
        # infer on the validation loader
        val_loss_avg = test("validation", model, val_loader, device)
        # store the average loss for this epoch
        val_ep_losses.append(val_loss_avg) 

    # return the list of epoch validation losses in order to use it later to create a plot
    return val_ep_losses

model = SiameseNetwork(input_dim=20, hidden_dim=300, n_classes=7)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else 'mps')
loss_list_val = run_epochs(model, args, optimizer, train_loader, device)

        # --------------------------------------------------- #
        # -------------- Plot validation loss --------------- #
        # --------------------------------------------------- #

def plot_loss(loss_list):
    '''
        Plots a simple curve showing the different values of the validation loss for each epoch.

        @param loss_list(list): A list of losses which length corresponds to the number of epochs
    '''
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('Epochs')
    # in our model we use Softmax then NLLLoss which means Cross Entropy loss
    plt.ylabel('Triplet loss')
    # in our training loop we used an Adam optimizer so we indicate it there
    plt.title('lr: {}, optim_alg:{}'.format(args['lr'], 'Adam'))
    # let's directly show the plot when calling this function
    plt.show()

plot_loss(loss_list_val)