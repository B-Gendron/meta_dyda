        # --------------------------------------------------- #
        # --------------------- Imports --------------------- #
        # --------------------------------------------------- #

# Torch utils
import torch
import torch.nn as nn

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
    
