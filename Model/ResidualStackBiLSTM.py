import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary


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

class ResidualStackBiLSTM(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(ResidualStackBiLSTM, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.ModuleList([])
        self.rnn_norm = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnn1.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_norm.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(input_size, output_size, 1),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        output = x.permute(0, 2, 1)
        for i in range(len(self.rnn1)):
            input = output 
            output = self.rnn1[i](input)
            output = output.permute(0, 2, 1)
            output = self.rnn_norm[i](output)
            output = output.permute(0, 2, 1)
            output = output + input 

        output = output.permute(0, 2, 1)
        output = self.output(output)

        return output
    
class ArousalModel_Physionet(nn.Module):
    def __init__(self, seq_len=5*60*200, input_size=8, hidden_size=64, output_size=1,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(ArousalModel_Physionet, self).__init__()

        self.arousal = ResidualStackBiLSTM(seq_len, input_size, hidden_size, output_size,
                 dropout, num_layers, bidirectional, repeat_times)

    def forward(self, x):
        output = self.arousal(x)

        return output

class ArousalApneaModel(nn.Module):
    def __init__(self, seq_len=5*60*200, input_size=8, hidden_size=64, output_size=1,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(ArousalApneaModel, self).__init__()

        self.arousal = ResidualStackBiLSTM(seq_len, input_size, hidden_size, output_size,
                 dropout, num_layers, bidirectional, repeat_times)
        
        self.apnea = ResidualStackBiLSTM(seq_len, input_size, hidden_size, output_size,
                 dropout, num_layers, bidirectional, repeat_times)

    def forward(self, x):
        outputArousal = self.arousal(x)
        outputApnea = self.apnea(x)

        return outputArousal, outputApnea

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualStackBiLSTM(seq_len=5*60*200, input_size=8, hidden_size=64, output_size=1, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*200), device=DEVICE)