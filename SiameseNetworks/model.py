        # --------------------------------------------------- #
        # --------------------- Imports --------------------- #
        # --------------------------------------------------- #

# Torch utils
import torch
import torch.nn as nn

# Data loading and preprocessing
from torchtext.vocab import FastText

        # --------------------------------------------------- #
        # ------------------- Model class ------------------- #
        # --------------------------------------------------- #

pretrained_vectors = FastText(language='en')

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