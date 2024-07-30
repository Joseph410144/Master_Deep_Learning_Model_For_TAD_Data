import torch
import torch.nn as nn
import torch.nn.functional as F
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
import numpy as np
from torch.nn.parallel import DataParallel

class RNNLayer(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, seq_len, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(RNNLayer, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = bidirectional

        self.rnn = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional= self.num_direction)

        # linear projection layer
        if self.num_direction:
            self.proj = nn.Linear(hidden_size * 2, input_size)
        else:
            self.proj = nn.Linear(hidden_size, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output.contiguous().view(-1, rnn_output.shape[2])).view(output.shape)
        return rnn_output

class DPRNN(nn.Module):
    """
    Deep duaL-path RNN.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 3):
        super(DPRNN, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.ModuleList([])
        self.rnn2 = nn.ModuleList([])
        self.rnn_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn2.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(input_size, output_size, 1),
                                    nn.Sigmoid()
                                    )


    def forward(self, x):
        # input shape: batch, C, dim1
        # apply RNN on dim1 twice
        # output shape: B, output_size, dim1
        # output = x.reshape((x.size(0), x.size(2), x.size(1)))
        output = x.permute(0, 2, 1)
        for i in range(len(self.rnn1)):
            input_rnn1 = output
            output_rnn1 = self.rnn1[i](input_rnn1)

            output_rnn1 = output_rnn1.permute(0, 2, 1)

            output_rnn1 = self.rnn_norm[i](output_rnn1)

            output_rnn1 = output_rnn1.permute(0, 2, 1)

            output_rnn1 = output_rnn1 + input_rnn1

            output_rnn2 = self.rnn2[i](output_rnn1)

            output_rnn2 = output_rnn2.permute(0, 2, 1)

            output_rnn2 = self.rnn_norm[i](output_rnn2)

            output_rnn2 = output_rnn2.permute(0, 2, 1)

            output = output_rnn2 + output_rnn1

        output = output.permute(0, 2, 1)
        output = self.output(output)

        return output

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

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

def FFT_for_Period_Channel(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    """each batch and each channel will get average frequency"""
    frequency_list_allChannel = abs(xf).mean(0)
    frequency_weight = abs(xf).mean(0)[:k, :]
    period_all = []
    top_list_all = []
    for ch in range(0, 7):
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

class TimesBlock(nn.Module):
    def __init__(self, seq_len=5*60*100, pred_len=0, top_k=5, d_model=8, d_ff=32, num_kernels=3, num_class=1, n_features=8):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.num_class = num_class
        self.n_features = n_features
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
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
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        res = res.permute(0, 2, 1)

        return res
    
class TimesBlock_FFTMod(nn.Module):
    def __init__(self, seq_len=5*60*100, pred_len=0, top_k=3, d_model=8, d_ff=32, num_kernels=3, num_class=1, n_features=8):
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
        period_list, period_weight = FFT_for_Period_Channel(x, self.k)

        res = []
        for i in range(self.k*7):
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
        res = res.permute(0, 2, 1)

        return res

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, pool_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(pool_size)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels+out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Sequential(
            # nn.Conv1d(out_channels+out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )

        # self.Point2Second = nn.Conv1d(30000, 300, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        # x = x.permute(0, 2, 1)
        # x = self.Point2Second(x)
        # x = x.permute(0, 2, 1)
        return x

class TimesUnet(nn.Module):
    """
    Channel = 8
    """
    def __init__(self, size=60*5*100, channels=8, num_class=1):
        super(TimesUnet, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 15, 2)
        self.conv0 = Down(15, 18, 2)
        self.conv1 = Down(18, 21, 2)
        self.conv2 = Down(21, 24, 2)

        self.connect1 = TimesBlock(seq_len=3750, pred_len=0, top_k=5, d_model=21, d_ff=42, 
                                                    num_kernels=3, num_class=21, n_features=21)
        # self.connect1 = TimesBlock_FFTMod(seq_len=3750, pred_len=0, top_k=3, d_model=21, d_ff=42, 
        #                                             num_kernels=3, num_class=21, n_features=21)
        self.up1 = Up(24, 21)
        self.connect2 = TimesBlock(seq_len=7500, pred_len=0, top_k=5, d_model=18, d_ff=36, 
                                                    num_kernels=3, num_class=18, n_features=18)
        # self.connect2 = TimesBlock_FFTMod(seq_len=7500, pred_len=0, top_k=3, d_model=18, d_ff=36, 
        #                                             num_kernels=3, num_class=18, n_features=18)
        self.up2 = Up(21, 18)
        self.connect3 = TimesBlock(seq_len=15000, pred_len=0, top_k=5, d_model=15, d_ff=30, 
                                                    num_kernels=3, num_class=15, n_features=15)
        # self.connect3 = TimesBlock_FFTMod(seq_len=15000, pred_len=0, top_k=3, d_model=15, d_ff=30, 
        #                                             num_kernels=3, num_class=15, n_features=15)
        self.up3 = Up(18, 15)

        self.last = OutConv(15, self.channels)

        self.Arousalout = DPRNN(seq_len=self.size, input_size=self.channels, hidden_size=64, output_size=self.num_class, dropout=0, 
                                num_layers=1, bidirectional=True, repeat_times = 3)
        self.Apneaout = DPRNN(seq_len=self.size, input_size=self.channels, hidden_size=64, output_size=self.num_class, dropout=0, 
                                num_layers=1, bidirectional=True, repeat_times = 3)
    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x3 = self.connect1(x3)
        x = self.up1(x4, x3)
        x2 = self.connect2(x2)
        x = self.up2(x, x2)
        x1 = self.connect3(x1)
        x = self.up3(x, x1)

        # channel 15 >> 8
        x = self.last(x)


        x_arousal = self.Arousalout(x)
        x_apnea = self.Apneaout(x)

        return x_arousal, x_apnea

if __name__ == "__main__":
    DEVICE = "cpu"
    # net = TimesBlock(seq_len=30000, pred_len=0, top_k=5, d_model=8, d_ff=32, 
    #                                                 num_kernels=3, num_class=5, n_features=5)
    net = TimesUnet(size=60*5*100, channels=8, num_class=1)
    # net = DPRNN(seq_len=60*5*100, input_size=8, hidden_size=64, output_size=1, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3, emb_dimension=128)
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*100), device=DEVICE)