import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel
from UnetModel.ConvTimeNet.ConvTimeNetBackBone import ConvTimeNet_backbone
# from UnetModel.Detr1D import Transformer1d

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

class SelfAtten(nn.Module):
    def __init__(self, dim, keyChannel):
        super(SelfAtten, self).__init__()
        self.d = dim
        self.keyCh = keyChannel
        self.Wq = nn.Conv1d(self.d, self.keyCh, kernel_size=1)
        self.Wk = nn.Conv1d(self.d, self.keyCh, kernel_size=1)
        self.Wv = nn.Conv1d(self.d, self.d, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, x):
        query = self.Wq(x).permute(0, 2, 1)  # Shape: (batch_size, key_channels, seq_len)
        key = self.Wk(x)  # Shape: (batch_size, key_channels, seq_len)
        value = self.Wv(x)  # Shape: (batch_size, value_channels, seq_len)
        energy = torch.bmm(query, key)  # Shape: (batch_size, seq_len, seq_len)
        attention = self.softmax(energy)  # Shape: (batch_size, seq_len, seq_len)
        output = torch.bmm(value, attention.permute(0, 2, 1))  # Shape: (batch_size, value_channels, seq_len)
        output = output.permute(0, 1, 2)  # Shape: (batch_size, seq_len, value_channels)

        return output

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

        # self.conv = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
        #     nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
        # )

    def forward(self, x, x_o):
        x = self.up(x)
        # x = torch.cat((x, x_o), 1)
        return self.conv(x)

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

""" Recommend from advisor Kuo """
class RNNLayer_ConV(nn.Module):
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
        super(RNNLayer_ConV, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = bidirectional

        self.rnn = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional= self.num_direction)

        # linear projection layer
        if self.num_direction:
            self.proj = nn.Conv1d(hidden_size * 2, input_size, kernel_size=1, stride=1)
        else:
            self.proj = nn.Conv1d(hidden_size, input_size, kernel_size=1, stride=1)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = rnn_output.permute(0, 2, 1)
        rnn_output = self.proj(rnn_output)
        rnn_output = rnn_output.permute(0, 2, 1)
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
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 3, emb_dimension=128):
        super(DPRNN, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.emb_dimension = emb_dimension
        
        """ 無作用，但是之前訓練模型時沒有註解掉，所以有時需要用到或是註解看之前是什麼情況 """
        self.pos_embd_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.emb_dimension, self.input_size),
        )

        self.rnn1 = nn.ModuleList([])
        self.rnn2 = nn.ModuleList([])
        self.rnn_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn2.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            """ Change Linear to ConV1x1 """
            # self.rnn1.append(RNNLayer_ConV(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            # self.rnn2.append(RNNLayer_ConV(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(input_size, output_size, 1),
                                    nn.Sigmoid()
                                    )

    def pos_encoding(self, t, Time_Dimension):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        pos_enc_all = torch.zeros(t.shape[0], 128).to(DEVICE)
        for index in range(0, t.shape[0]):
            t_emb = t[index][0]
            t_emb = torch.tensor([t_emb])
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, Time_Dimension, 2).float() / Time_Dimension)
            )
            pos_enc_a = torch.sin(t_emb.repeat(1, Time_Dimension // 2) * inv_freq)
            pos_enc_b = torch.cos(t_emb.repeat(1, Time_Dimension // 2) * inv_freq)
            pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
            pos_enc_all[index, :] = pos_enc[0, :]
        return pos_enc_all

    def forward(self, x):
        # input shape: batch, C, dim1
        # apply RNN on dim1 twice
        # output shape: B, output_size, dim1
        # output = x.reshape((x.size(0), x.size(2), x.size(1)))
        output = x.permute(0, 2, 1)
        # pos = self.pos_encoding(t, self.emb_dimension)
        # t_emb = self.pos_embd_layer(pos)[:, :, None].repeat(1, 1, self.seq_len).permute(0, 2, 1)
        for i in range(len(self.rnn1)):
            input_rnn1 = output #+ t_emb
            output_rnn1 = self.rnn1[i](input_rnn1)

            # output_rnn1 = output_rnn1.reshape((output_rnn1.size(0), output_rnn1.size(2), output_rnn1.size(1)))
            output_rnn1 = output_rnn1.permute(0, 2, 1)

            output_rnn1 = self.rnn_norm[i](output_rnn1)

            # output_rnn1 = output_rnn1.reshape((output_rnn1.size(0), output_rnn1.size(2), output_rnn1.size(1)))
            output_rnn1 = output_rnn1.permute(0, 2, 1)

            output_rnn1 = output_rnn1 + input_rnn1 #+ t_emb

            output_rnn2 = self.rnn2[i](output_rnn1)

            # output_rnn2 = output_rnn2.reshape((output_rnn2.size(0), output_rnn2.size(2), output_rnn2.size(1)))
            output_rnn2 = output_rnn2.permute(0, 2, 1)

            output_rnn2 = self.rnn_norm[i](output_rnn2)

            # output_rnn2 = output_rnn2.reshape((output_rnn2.size(0), output_rnn2.size(2), output_rnn2.size(1)))
            output_rnn2 = output_rnn2.permute(0, 2, 1)

            output = output_rnn2 + output_rnn1

        # output = output.reshape((output.size(0), output.size(2), output.size(1)))
        output = output.permute(0, 2, 1)
        output = self.output(output)

        return output
    
class DPRNN_2D(nn.Module):
    """
    Deep duaL-path RNN 2D.

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
        super(DPRNN_2D, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.rnn1 = nn.ModuleList([])
        self.rnn2 = nn.ModuleList([])
        self.rnn1_norm = nn.ModuleList([])
        self.rnn2_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn2.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn1_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))
            self.rnn2_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(input_size, output_size, 1),
                                    nn.Sigmoid()
                                    )
    def SignalSegAndOvp(self, series, batch, device):
        pad = torch.nn.ZeroPad2d(padding=(25, 25, 0, 0))
        overlapped_series = torch.zeros(batch, 8, 200, 200).to(device)
        series_test = pad(series)
        for i in range(0, 200):
            overlapped_series[:, :, i, :] = series_test[:, :, i*150:(i*150)+200]
        return overlapped_series

    def SignalReduction(self, series, batch, device):
        original_series = torch.zeros(batch, 8, 200, 150).to(device)
        for i in range(0, 200):
            original_series[:, :, i, :] = series[:, :, i, 25:175]

        original_series = original_series.view(batch, 8, 200*150).contiguous()
        return original_series

    def forward(self, x):
        """
        input shape: batch, C, dim
        reshape input: batch, C, dim1, dim2
        apply RNN on dim1 twice
        output shape: B, output_size, dim
        """
        # output: (b, 8, 150, 200)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size, _, dim1 = x.shape
        output = self.SignalSegAndOvp(x, batch_size, DEVICE)
        batch_size, _, dim1, dim2 = output.shape
        for i in range(len(self.rnn1)):
            input_rnn1 = output
            input_rnn1 = input_rnn1.permute(0, 3, 2, 1).contiguous().view(batch_size * dim2, dim1, -1)  # B*dim2, dim1, N
            output_rnn1 = self.rnn1[i](input_rnn1)
            output_rnn1 = output_rnn1.view(batch_size, dim2, dim1, -1).permute(0, 3, 2, 1).contiguous()  # B, N, dim1, dim2
            output_rnn1 = self.rnn1_norm[i](output_rnn1)
            output_rnn1 = output_rnn1 + output

            input_rnn2 = output_rnn1
            input_rnn2 = input_rnn2.permute(0, 2, 3, 1).contiguous().view(batch_size * dim1, dim2, -1)  # B*dim1, dim2, N
            output_rnn2 = self.rnn1[i](input_rnn2)
            output_rnn2 = output_rnn2.view(batch_size, dim1, dim2, -1).permute(0, 3, 1, 2).contiguous()  # B, N, dim1, dim2
            output_rnn2 = self.rnn1_norm[i](output_rnn2)
            output = output_rnn2 + output_rnn1

        # output = output.view(batch_size, self.input_size, dim1*dim2).contiguous()
        
        output = self.SignalReduction(output, batch_size, DEVICE)
        output = self.output(output)

        return output

class USleep_5min_Encoder(nn.Module):
    """
    Channel = 8
    """
    def __init__(self, size=60*5*100, channels=5, num_class=1):
        super(USleep_5min_Encoder, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 15, 2)
        self.conv0 = Down(15, 18, 2) 
        self.conv1 = Down(18, 21, 2)
        self.conv2 = Down(21, 24, 2)
        self.up1 = Up(24, 21) # 15
        self.up2 = Up(21, 18)
        self.up3 = Up(18, 15)
        self.last = OutConv(15, channels)

    def forward(self, x):
        x_o = x
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        # x5 = self.conv3(x4)
        # x = self.up0(x5, x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.last(x, x_o)
        return output
    
class USleep_5min_Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=1, n_features=1, bi=False):
        super(USleep_5min_Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 64, n_features
        self.bi = bi
        self.rnn1 = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
            )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bi # if use bilstm >> hidden_dim need to be hidden_dim*2
            )
        if self.bi:
            self.output_layer = nn.Linear(self.hidden_dim*2, self.n_features)
        else:
            self.output_layer = nn.Linear(self.hidden_dim, self.n_features)

        # self.Linear_layer = nn.Sequential(
        #     nn.Linear(self.seq_len, self.seq_len//100)
        # )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):

        batch = x.size(0)
        # x = torch.transpose(x, 1, 2)
        x = x.reshape((batch, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        if self.bi:
            x = x.reshape((batch, self.seq_len, self.hidden_dim*2))
        else:
            x = x.reshape((batch, self.seq_len, self.hidden_dim))
        x = self.output_layer(x)
        # x = torch.transpose(x, 2, 1)
        x = x.reshape((batch, self.n_features, self.seq_len))
        
        return self.Sigmoid(x)

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

def ShortTimeFFT(x):
    B, C, T = x.shape
    seg = T//(30*100)
    for i in range(0, seg):
        xf = torch.fft.rfft(x, dim=2)
        xf = xf[:, :, 1:]


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

        self.classifier = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.n_features, self.num_class),
            nn.Sigmoid()
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
        # period_weight = F.softmax(period_weight, dim=1)
        # period_weight = period_weight.unsqueeze(
        #     1).unsqueeze(1).repeat(1, T, N, 1)
        # res = torch.sum(res * period_weight, -1)
        res = torch.sum(res * 1, -1)
        # residual connection
        res = res + x
        """ for feature extractor """
        # res = self.classifier(res)
        # res = res.permute(0, 2, 1)
        return res

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

        return res
    

class ArousalApneaUENNModel(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(ArousalApneaUENNModel, self).__init__()
        """ Encoder : Unet """
        self.encoder = USleep_5min_Encoder(size, n_features, num_class)

        """ Encoder : TimesNet """
        # self.layer_norm = nn.LayerNorm(8)
        # # self.encoder = nn.ModuleList([TimesBlock(seq_len=size, pred_len=0, top_k=5, d_model=8, d_ff=32, num_kernels=3, num_class=num_class, n_features=n_features)
        # #                             for _ in range(2)])
        # self.encoder = nn.ModuleList([TimesBlock_FFTMod(seq_len=size, pred_len=0, top_k=5, d_model=8, d_ff=32, num_kernels=3, num_class=num_class, n_features=n_features)
        #                             for _ in range(2)])

        self.encoder = ConvTimeNet_backbone(c_in=8, seq_len=int(5*60*100), context_window = int(5*60*100),
                                target_window=5*60*100, patch_len=50, stride=160, n_layers=6, d_model=64, d_ff=256, dw_ks=[9,11,15,21,29,39], norm="batch", dropout=0.0, act="gelu", head_dropout=0.0, padding_patch = None, head_type="flatten", 
                                revin=1, affine=0, deformable=True, subtract_last=0, enable_res_param=1, re_param=1, re_param_kernel=3)


        """Decoder : LSTM stacks"""
        # self.decoderArousal = USleep_5min_Decoder(size, 8, num_class, False)
        # self.decoderApnea = USleep_5min_Decoder(size, 8, num_class, False)

        """ Decoder : DPRNN """
        self.decoderArousal = DPRNN(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3, emb_dimension=128)
        self.decoderApnea = DPRNN(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3, emb_dimension=128)

        """ Decoder : DPRNN_2D """
        # self.decoderArousal = DPRNN_2D(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
        # self.decoderApnea = DPRNN_2D(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)

        """ Decoder : Transformer1d """
        # self.decoderArousal = Transformer1d(n_classes=1, 
        #                         n_length=3*60*100, 
        #                         d_model=8, 
        #                         nhead=1, 
        #                         dim_feedforward=128, 
        #                         dropout=0.1, 
        #                         activation='relu',
        #                         verbose = True)

        # self.decoderApnea = Transformer1d(n_classes=1, 
        #                         n_length=3*60*100, 
        #                         d_model=8, 
        #                         nhead=1, 
        #                         dim_feedforward=128, 
        #                         dropout=0.1, 
        #                         activation='relu',
        #                         verbose = True)

        

    def forward(self, x):
        x_encoder = x
        # for i in range(2):
        #     x_encoder = self.layer_norm(self.encoder[i](x_encoder))
        #     x_encoder = x_encoder.permute(0, 2, 1)
        
        x_encoder = self.encoder(x_encoder)
        x_arousal = self.decoderArousal(x_encoder)
        x_apnea = self.decoderApnea(x_encoder)
        return x_arousal, x_apnea#, x_encoder


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # net = TimesBlock()
    net = ArousalApneaUENNModel(size=5*60*100, num_class=1, n_features=8)
    # net = DPRNN_2D(seq_len=5*60*100, input_size=8, hidden_size=64, output_size=1, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*100), device=DEVICE)