import torch
import torch.nn as nn
import torch.nn.functional as F
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel



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
    

class USleep_5min_Encoder(nn.Module):
    """
    Channel = 6
    """
    def __init__(self, size=60*5*100, channels=5, num_class=1):
        super(USleep_5min_Encoder, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        # self.conv01 = Down(self.channels, 9, 2)
        # self.conv0 = Down(9, 12, 2) 
        # self.conv1 = Down(12, 15, 2)
        # self.conv2 = Down(15, 18, 2)
        # # self.conv3 = Down(18, 21, 2)
        # # self.up0 = Up(21, 18) # 18+18
        # self.up1 = Up(18, 15) # 15
        # self.up2 = Up(15, 12)
        # self.up3 = Up(12, 9)
        # self.last = OutConv(9, self.channels)
        self.conv01 = Down(self.channels, 15, 2)
        self.conv0 = Down(15, 18, 2) 
        self.conv1 = Down(18, 21, 2)
        self.conv2 = Down(21, 24, 2)
        # self.conv3 = Down(18, 21, 2)
        # self.up0 = Up(21, 18) # 18+18
        self.up1 = Up(24, 21) # 15
        self.up2 = Up(21, 18)
        self.up3 = Up(18, 15)
        self.last = OutConv(15, self.channels)

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
        # x = x.repeat(1, self.seq_len, self.n_features)
        # x = x.reshape((self.seq_len*2, self.n_features))
        batch = x.size(0)
        x = torch.transpose(x, 1, 2)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        if self.bi:
            x = x.reshape((batch, self.seq_len, self.hidden_dim*2))
        else:
            x = x.reshape((batch, self.seq_len, self.hidden_dim))
        x = self.output_layer(x)
        x = torch.transpose(x, 2, 1)
        # x = self.Linear_layer(x)
        
        return self.Sigmoid(x)

class ULstmAutoencoder_5min(nn.Module):
    def __init__(self, size, num_class, n_features):
        super(ULstmAutoencoder_5min, self).__init__()
        self.encoder = USleep_5min_Encoder(size, n_features, num_class)
        self.decoder = USleep_5min_Decoder(size, n_features, num_class, True)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = ULstmAutoencoder_5min(size=5*60*100, num_class=1, n_features=5)
    net.to(DEVICE)
    # if torch.cuda.device_count() > 1:
    #     net = DataParallel(net)
    # results = summary(net, input_size=(16, 6, 5*60*100), device=DEVICE)
    with open('ULstmAutoencoder.log', 'w') as f:
        report = summary(net, input_size=(16, 5, 5*60*100), device=DEVICE)
        f.write(str(report))
    # net = USleep_5min(size=5*60*200, channels=6, num_class=1)
    # net.to(DEVICE)
    # summary(net, (6, 5*60*200), device=DEVICE)