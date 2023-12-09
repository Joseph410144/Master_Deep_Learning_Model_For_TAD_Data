import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchsummary import summary


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

class FC_Layer(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels):
        super(FC_Layer, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.Sigmoid()
        )   
    def forward(self, x):
        x = self.conv(x)
        return self.fc_layer(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)
    

class USleep_1min(nn.Module):
    """
    Channel = 6
    """
    def __init__(self, size=4096*1024, channels=13, num_class=7):
        super(USleep_1min, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 12, 2)
        self.conv0 = Down(12, 18, 2) 
        self.conv1 = Down(18, 21, 2)
        self.conv2 = Down(21, 24, 2)
        self.up5 = Up(24, 21) # 21+21
        self.up6 = Up(21, 18) # 18+18
        self.up7 = Up(18, 12) # 15
        self.last = OutConv(12, self.num_class)

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x = self.up5(x4, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        output = self.last(x)
        return output
    


class USleep_1minFCLayer(nn.Module):
    """
    Channel = 6
    """
    def __init__(self, size=60*200, channels=6, num_class=1):
        super(USleep_1min, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 12, 2)
        self.conv0 = Down(12, 18, 2) 
        self.conv1 = Down(18, 21, 2)
        self.conv2 = Down(21, 24, 2)
        self.up5 = Up(24, 21) # 21+21
        self.up6 = Up(21, 18) # 18+18
        self.up7 = Up(18, 12) # 15
        self.last = FC_Layer(self.size, self.size//200, 12, self.num_class)
        
        

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x = self.up5(x4, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        output = self.last(x)
        return output


class USleepAtten_1min(nn.Module):
    """
    Channel = 6
    """
    def __init__(self, size=200*60, channels=6, num_class=1):
        super(USleepAtten_1min, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 15, 2)
        self.atten01 = SelfAtten(15, 15)
        self.conv0 = Down(15, 18, 2) 
        self.atten0 = SelfAtten(18, 18)
        self.conv1 = Down(18, 21, 2)
        self.atten1 = SelfAtten(21, 21)
        self.conv2 = Down(21, 25, 2)
        self.atten2 = SelfAtten(25, 25)
        self.up5 = Up(25, 21) # 21+21
        self.atten3 = SelfAtten(21, 21)
        self.up6 = Up(21, 18) # 18+18
        self.atten4 = SelfAtten(18, 18)
        self.up7 = Up(18, 15) # 15
        self.atten5 = SelfAtten(15, 15)
        self.last = OutConv(15, self.num_class)

    def forward(self, x):
        x1 = self.conv01(x)
        x11 = self.atten01(x1)
        x2 = self.conv0(x11)
        x22 = self.atten0(x2)
        x3 = self.conv1(x22)
        x33 = self.atten1(x3)
        x4 = self.conv2(x33)
        x44 = self.atten2(x4)
        x = self.up5(x44, x33)
        x = self.atten3(x)
        x = self.up6(x, x22)
        x = self.atten4(x)
        x = self.up7(x, x11)
        x = self.atten5(x)
        output = self.last(x)
        return output


class USleep_5min(nn.Module):
    """
    Channel = 6
    """
    def __init__(self, size=60*5*200, channels=6, num_class=1):
        super(USleep_5min, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 9, 2)
        self.conv0 = Down(9, 12, 2) 
        self.conv1 = Down(12, 15, 2)
        self.conv2 = Down(15, 18, 2)
        self.conv3 = Down(18, 21, 2)
        self.up0 = Up(21, 18) # 18+18
        self.up1 = Up(18, 15) # 15
        self.up2 = Up(15, 12)
        self.up3 = Up(12, 9)
        self.last = OutConv(9, self.num_class)
        

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x5 = self.conv3(x4)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.last(x)
        return output
    
class USleep_5minFC_Layer(nn.Module):
    """
    Channel = 6
    """
    def __init__(self, size=60*5*200, channels=6, num_class=1):
        super(USleep_5min, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 9, 2)
        self.conv0 = Down(9, 12, 2) 
        self.conv1 = Down(12, 15, 2)
        self.conv2 = Down(15, 18, 2)
        self.conv3 = Down(18, 21, 2)
        self.up0 = Up(21, 18) # 18+18
        self.up1 = Up(18, 15) # 15
        self.up2 = Up(15, 12)
        self.up3 = Up(12, 9)
        self.last = FC_Layer(self.size, self.size//200, 9, self.num_class)

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x5 = self.conv3(x4)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.last(x)


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # net = USleep_5min(size=5*60*200, channels=6, num_class=1)
    # net.to(DEVICE)
    # summary(net, (6, 5*60*200), device=DEVICE)

    net = USleep_5min(size=5*60*200, channels=5, num_class=1)
    net.to(DEVICE)
    summary(net, (5, 5*60*200), device=DEVICE)
    
    
        
