import torch
import torch.nn as nn
import torch.nn.functional as F
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
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
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 3, emb_dimension=128):
        super(DPRNN, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.emb_dimension = emb_dimension
        
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

class DPRNNClassifier(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(DPRNNClassifier, self).__init__()
        self.classifierArousal = DPRNN(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3, emb_dimension=128)
        self.classifierApnea = DPRNN(size, n_features, 64, num_class, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3, emb_dimension=128)

        

    def forward(self, x):
        x_arousal = self.classifierArousal(x)
        x_apnea = self.classifierApnea(x)
        return x_arousal, x_apnea


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = DPRNNClassifier(size=5*60*100, num_class=1, n_features=8)
    # net = DPRNN_2D(seq_len=5*60*100, input_size=8, hidden_size=64, output_size=1, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*100), device=DEVICE)