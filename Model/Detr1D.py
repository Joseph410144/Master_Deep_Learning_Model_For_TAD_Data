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

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return self.data.shape[0]
    
class Transformer1d(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples, n_length) 
    Output:
        out: (1, n_length)
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.dense = nn.Linear(self.d_model, self.n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        out = x
        out = out.permute(2, 0, 1)
        out = self.transformer_encoder(out)
        out = out.reshape(out.size(1), out.size(0), out.size(2))
        out = self.dense(out)
        out = out.reshape(out.size(0), out.size(2), out.size(1))
        out = self.sigmoid(out)
        # out = self.softmax(out)
        
        return out    


if __name__ == '__main__':
    DEVICE = "cpu"
    net = Transformer1d(n_classes=1, 
        n_length=30000, 
        d_model=8, 
        nhead=2, 
        dim_feedforward=128, 
        dropout=0.1, 
        activation='relu',
        verbose = True)
    # net = DPRNN(30000, 8, 64, 1, dropout=0, num_layers=1, bidirectional=True, repeat_times = 5)
    net.to(DEVICE)

    # block = Down(5, 15, 2)
    # with open('Down.log', 'w') as f:
    report = summary(net, input_size=(1, 8, 5*60*100), device=DEVICE)