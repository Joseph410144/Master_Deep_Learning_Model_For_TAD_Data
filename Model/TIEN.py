import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel
# from UnetModel.ConvTimeNet.ConvTimeNetBackBone import ConvTimeNet_backbone
# from UnetModel.Detr1D import Transformer1d

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

def FFT_for_Period_Channel(x, k=2, channelNum=8):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    """each batch and each channel will get average frequency"""
    frequency_list_allChannel = abs(xf).mean(0)
    frequency_weight = abs(xf).mean(0)[:k, :]
    period_all = []
    top_list_all = []
    for ch in range(0, channelNum):
        frequency_list = frequency_list_allChannel[:, ch]
        """  xf[0] is integral x[n] """
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        # frequency_weight[:, ch] = frequency_list_allChannel[top_list, ch]
        period = x.shape[1] // top_list
        for i in range(len(period)):
            if period[i] == 0:
                period[i] = 1
        period_all = period_all + list(period)
        top_list_all = top_list_all + list(top_list)
    
    frequency_weight = abs(xf).mean(0)[top_list_all, :]
    period_all = np.array(list(period_all))
    return period_all, frequency_weight

class TimesBlock_FFTMod(nn.Module):
    def __init__(self, seq_len=5*60*100, pred_len=0, top_k=5, d_model=8, d_ff=32, num_kernels=3, num_class=1, n_features=8):
        super(TimesBlock_FFTMod, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.num_class = num_class
        self.n_features = n_features
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(self.n_features, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, self.n_features,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period_Channel(x, self.k, self.n_features)

        res = []
        for i in range(self.k*self.n_features):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            # print(length, period, length // period)
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=0)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).permute(1, 2, 3, 0).repeat(B, T, 1, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x

        return res
    