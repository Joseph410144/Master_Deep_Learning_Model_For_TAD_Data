import torch
import torch.nn as nn
import torch.nn.functional as F
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel

class RNNLayer(nn.Module):
    """
    Container module for a single LSTM layer.

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

class LSTM_SingleTask(nn.Module):
    """
    Only Arousal or Apnea Task
    """

    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(LSTM_SingleTask, self).__init__()

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

        output = output.permute(0, 2, 1)
        output = self.output(output)

        return output
    
class LSTM_MultiTask(nn.Module):
    """
    Arousal, Apnea Task
    """

    def __init__(self, seq_len, input_size, hidden_size, output_size,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 6):
        super(LSTM_MultiTask, self).__init__()

        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnnArousal = nn.ModuleList([])
        self.rnn_normArousal = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnnArousal.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_normArousal.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.rnnApnea = nn.ModuleList([])
        self.rnn_normApnea = nn.ModuleList([])
        for i in range(repeat_times):
            self.rnnApnea.append(RNNLayer(seq_len, input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional))
            self.rnn_normApnea.append(nn.GroupNorm(1, input_size, eps=1e-8))

        self.outputArousal = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(input_size, output_size, 1),
                                    nn.Sigmoid()
                                    )
        
        self.outputApnea = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(input_size, output_size, 1),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        output = x.permute(0, 2, 1)
        for i in range(len(self.rnnArousal)):
            inputArousal = output 
            outputArousal = self.rnnArousal[i](inputArousal)
            outputArousal = outputArousal.permute(0, 2, 1)
            outputArousal = self.rnn_normArousal[i](outputArousal)
            outputArousal = outputArousal.permute(0, 2, 1)
        
        for i in range(len(self.rnnApnea)):
            inputApnea = output 
            outputApnea = self.rnnApnea[i](inputApnea)
            outputApnea = outputApnea.permute(0, 2, 1)
            outputApnea = self.rnn_normApnea[i](outputApnea)
            outputApnea = outputApnea.permute(0, 2, 1)

        outputArousal = outputArousal.permute(0, 2, 1)
        outputArousal = self.outputArousal(outputArousal)

        outputApnea = outputApnea.permute(0, 2, 1)
        outputApnea = self.outputApnea(outputApnea)


        return outputArousal, outputApnea
    

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = LSTM_SingleTask(seq_len=5*60*200, input_size=8, hidden_size=64, output_size=1,
                 dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
    # net = DPRNN_2D(seq_len=5*60*100, input_size=8, hidden_size=64, output_size=1, dropout=0, num_layers=1, bidirectional=True, repeat_times = 3)
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*200), device=DEVICE)