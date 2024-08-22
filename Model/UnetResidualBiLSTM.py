import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel
from Model.Unet import Unet
from Model.ResidualStackBiLSTM import ResidualStackBiLSTM
    
class ArousalApneaModel(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(ArousalApneaModel, self).__init__()
        """ Encoder : Unet """
        self.encoder = Unet(size=size, channels=n_features, num_class=num_class)


        """ Decoder : Residual Stacked Bi-LSTM """
        # hidden unit: 32, 64, 128
        hidden_unit = 64
        self.decoderArousal = ResidualStackBiLSTM(size, n_features, hidden_unit, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 6)
        self.decoderApnea = ResidualStackBiLSTM(size, n_features, hidden_unit, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 6)

    def forward(self, x):
        x_encoder = self.encoder(x)
        x_arousal = self.decoderArousal(x_encoder)
        x_apnea = self.decoderApnea(x_encoder)

        return x_arousal, x_apnea
    
class ArousalModel_Physionet(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(ArousalModel_Physionet, self).__init__()
        """ Encoder : Unet """
        self.encoder = Unet(size=size, channels=n_features, num_class=num_class)

        """ Decoder : Residual Stacked Bi-LSTM """
        # hidden unit: 32, 64, 128
        hidden_unit = 64
        self.decoderArousal = ResidualStackBiLSTM(size, n_features, hidden_unit, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 6)

    def forward(self, x):
        x_encoder = self.encoder(x)
        x_arousal = self.decoderArousal(x_encoder)
        
        return x_arousal


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = ArousalApneaModel(size=5*60*100, num_class=1, n_features=8)
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*100), device=DEVICE)