import torch
import torch.nn as nn
import torch.nn.functional as F
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel
# from UnetModel.Detr1D import Transformer1d



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
            """ Change Linear to ConV1x1 """
            # self.rnn1.append(RNNLayer_ConV(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            # self.rnn2.append(RNNLayer_ConV(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
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

            # output_rnn1 = output_rnn1.reshape((output_rnn1.size(0), output_rnn1.size(2), output_rnn1.size(1)))
            output_rnn1 = output_rnn1.permute(0, 2, 1)

            output_rnn1 = self.rnn_norm[i](output_rnn1)

            # output_rnn1 = output_rnn1.reshape((output_rnn1.size(0), output_rnn1.size(2), output_rnn1.size(1)))
            output_rnn1 = output_rnn1.permute(0, 2, 1)

            output_rnn1 = output_rnn1 + input_rnn1

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

    def forward(self, x):
        """
        input shape: batch, C, dim
        reshape input: batch, C, dim1, dim2
        apply RNN on dim1 twice
        output shape: B, output_size, dim
        """
        # output: (b, 8, 150, 200)
        output = x.reshape((x.size(0), x.size(1), 150, 200))
        for i in range(len(self.rnn1)):

            input_rnn1 = output
            # input_rnn1: (b, 200, 8, 150)
            input_rnn1 = input_rnn1.permute(0, 3, 1, 2)
            # input_rnn1: (b*200, 8, 150)
            input_rnn1 = input_rnn1.reshape((input_rnn1.size(0)*input_rnn1.size(1), input_rnn1.size(2), input_rnn1.size(3)))
            # input_rnn1: (b*200, 150, 8)
            input_rnn1 = input_rnn1.permute(0, 2, 1)
            # output_rnn1: (b*200, 150, 8)
            output_rnn1 = self.rnn1[i](input_rnn1)
            # output_rnn1: (b*200, 8, 150)
            output_rnn1 = output_rnn1.permute(0, 2, 1)
            # output_rnn1: (b*200, 8, 150)
            output_rnn1 = self.rnn1_norm[i](output_rnn1)
            # output_rnn1: (b, 200, 8, 150)
            output_rnn1 = output_rnn1.reshape((output_rnn1.size(0)//200, 200, output_rnn1.size(1), output_rnn1.size(2)))
            # output_rnn1: (b, 8, 150, 200)
            output_rnn1 = output_rnn1.permute(0, 2, 3, 1)
            # output_rnn1: (b, 8, 150, 200)
            output_rnn1 = output_rnn1 + output

            input_rnn2 = output_rnn1

            # input_rnn2: (b, 150, 8, 200)
            input_rnn2 = input_rnn2.permute(0, 2, 1, 3)
            # input_rnn2: (b*150, 8, 200)
            input_rnn2 = input_rnn2.reshape((input_rnn2.size(0)*input_rnn2.size(1), input_rnn2.size(2), input_rnn2.size(3)))
            # input_rnn2: (b*150, 200, 8)
            input_rnn2 = input_rnn2.permute(0, 2, 1)
            # output_rnn2: (b*150, 200, 8)
            output_rnn2 = self.rnn1[i](input_rnn2)
            # output_rnn2: (b*150, 8, 200)
            output_rnn2 = output_rnn2.permute(0, 2, 1)
            # output_rnn2: (b*150, 8, 200)
            output_rnn2 = self.rnn1_norm[i](output_rnn2)
            # output_rnn2: (b, 150, 8, 200)
            output_rnn2 = output_rnn2.reshape((output_rnn2.size(0)//150, 150, output_rnn2.size(1), output_rnn2.size(2)))
            # output_rnn2: (b, 8, 150, 200)
            output_rnn2 = output_rnn2.permute(0, 2, 1, 3)
            # output_rnn2: (b, 8, 150, 200)
            output = output_rnn2 + output_rnn1


        output = output.reshape((output.size(0), output.size(1), output.size(2)*output.size(3)))
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
        self.last = OutConv(15, 8)

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

class ArousalApneaUENNModel(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(ArousalApneaUENNModel, self).__init__()
        self.encoder = USleep_5min_Encoder(size, n_features, num_class)

        """Decoder : LSTM stacks"""
        # self.decoderArousal = USleep_5min_Decoder(size, 8, num_class, False)
        # self.decoderApnea = USleep_5min_Decoder(size, 8, num_class, False)

        """ Decoder : DPRNN """
        # self.decoderArousal = DPRNN(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
        # self.decoderApnea = DPRNN(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)

        """ Decoder : DPRNN_2D """
        self.decoderArousal = DPRNN_2D(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
        self.decoderApnea = DPRNN_2D(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)

        """ Decoder : Transformer1D """
        # self.decoderArousal = Transformer1d(n_classes=num_class, 
        #                                     n_length=size, 
        #                                     d_model=n_features, 
        #                                     nhead=2, 
        #                                     dim_feedforward=64, 
        #                                     dropout=0.1, 
        #                                     activation='relu',
        #                                     verbose = True)
        # self.decoderApnea = Transformer1d(n_classes=num_class, 
        #                                     n_length=size, 
        #                                     d_model=n_features, 
        #                                     nhead=2, 
        #                                     dim_feedforward=64, 
        #                                     dropout=0.1, 
        #                                     activation='relu',
        #                                     verbose = True)
        

    def forward(self, x):
        x_encoder = self.encoder(x)
        x_arousal = self.decoderArousal(x_encoder)
        x_apnea = self.decoderApnea(x_encoder)
        return x_arousal, x_apnea


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = ArousalApneaUENNModel(size=5*60*100, num_class=1, n_features=8)
    net.to(DEVICE)

    report = summary(net, input_size=(8, 8, 5*60*100), device=DEVICE)