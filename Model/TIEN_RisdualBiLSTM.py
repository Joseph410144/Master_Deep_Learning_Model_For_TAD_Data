import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchinfo import summary
from torch.nn.parallel import DataParallel
from Model.ResidualStackBiLSTM import ResidualStackBiLSTM
from Model.TIEN import TimesBlock_FFTMod

class ArousalApneaModel(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(ArousalApneaModel, self).__init__()

        """ Encoder : TimesNet """
        self.layer_norm = nn.LayerNorm(n_features)
        self.encoder = nn.ModuleList([TimesBlock_FFTMod(seq_len=size, pred_len=0, top_k=2, d_model=n_features, d_ff=32, num_kernels=3, num_class=num_class, n_features=n_features)
                                    for _ in range(2)])

        """ Decoder : Residual Stacked Bi-LSTM """
        # hidden unit: 32, 64, 128
        hidden_unit = 64
        self.decoderArousal = ResidualStackBiLSTM(size, n_features, hidden_unit, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 6)
        self.decoderApnea = ResidualStackBiLSTM(size, n_features, hidden_unit, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 6)

    def forward(self, x):
        x_encoder = x
        for i in range(2):
            x_encoder = self.layer_norm(self.encoder[i](x_encoder))
            x_encoder = x_encoder.permute(0, 2, 1)
        
        x_arousal = self.decoderArousal(x_encoder)
        x_apnea = self.decoderApnea(x_encoder)
        return x_arousal, x_apnea

class ArousalApneaModel_Physionet(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(ArousalApneaModel_Physionet, self).__init__()

        """ Encoder : TimesNet """
        self.layer_norm = nn.LayerNorm(n_features)
        self.encoder = nn.ModuleList([TimesBlock_FFTMod(seq_len=size, pred_len=0, top_k=2, d_model=n_features, d_ff=32, num_kernels=3, num_class=num_class, n_features=n_features)
                                    for _ in range(2)])

        """ Decoder : Residual Stacked Bi-LSTM """
        # hidden unit: 32, 64, 128
        hidden_unit = 64
        self.decoderArousal = ResidualStackBiLSTM(size, n_features, hidden_unit, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 6)

    def forward(self, x):
        x_encoder = x
        for i in range(2):
            x_encoder = self.layer_norm(self.encoder[i](x_encoder))
            x_encoder = x_encoder.permute(0, 2, 1)
        
        x_arousal = self.decoderArousal(x_encoder)
        return x_arousal

if __name__ == '__main__':
    DEVICE = "cpu"
    net = ArousalApneaModel(size=5*60*200, num_class=1, n_features=8)
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*200), device=DEVICE)