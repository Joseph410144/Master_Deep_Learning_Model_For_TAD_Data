import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchinfo import summary
from Model import TIEN_RisdualBiLSTM, Unet, UnetResidualBiLSTM, ResidualStackBiLSTM

if __name__ == '__main__':
    DEVICE = "cpu"
    net = ResidualStackBiLSTM.ArousalApneaModel()
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*200), device=DEVICE)