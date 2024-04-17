"""
resnet for 1-d signal data, pytorch version

Shenda Hong, Nov 2019
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

    
class Transformer1d(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_classes)
        
    Pararmetes:
        
    """

    def __init__(self, n_classes, n_length, d_model, nhead, dim_feedforward, dropout, activation, verbose=False):
        super(Transformer1d, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.n_length = n_length
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.n_classes = n_classes
        self.verbose = verbose
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=self.dim_feedforward, 
            dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.dense = nn.Linear(self.d_model, self.n_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # B, C, T
        out = x
        # T, B, C
        out = out.permute(2, 0, 1)

        out = self.transformer_encoder(out)

        # B, T, C
        out = out.permute(1, 0, 2)

        out = self.dense(out)

        # B, C, T
        out = out.permute(0, 2, 1)

        out = self.sigmoid(out)
        
        return out    

if __name__ == "__main__":
    model = Transformer1d(n_classes=1, 
        n_length=5*60*100, 
        d_model=8, 
        nhead=4, 
        dim_feedforward=128, 
        dropout=0.1, 
        activation='relu',
        verbose = True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)
    summary(model, (8, 8, 5*60*100), device=DEVICE)