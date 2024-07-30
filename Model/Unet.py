import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
# from torchsummary import summary
from torchinfo import summary

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

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
    
class OutConvStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConvStage, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.conv(x)

class USleepStage(nn.Module):
    """
    Channel = 5
    """
    def __init__(self, size=4096*1024, channels=13, num_class=7):
        super(USleepStage, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 15, 2)
        self.conv0 = Down(15, 18, 4) 
        self.conv1 = Down(18, 21, 4)
        self.conv2 = Down(21, 25, 4)
        self.conv3 = Down(25, 30, 4)
        self.conv4 = Down(30, 60, 4)
        self.conv5 = Down(60, 120, 4)
        self.conv6 = Down(120, 240, 4)
        self.conv7 = Down(240, 480, 4)
        self.up0 = Up(480, 240) # 240+240
        self.up1 = Up(240, 120) # 120+120
        self.up2 = Up(120, 60) # 60+60
        self.up3 = Up(60, 30) # 30+30
        self.up4 = Up(30, 25) # 25+25
        self.up5 = Up(25, 21) # 21+21
        self.up6 = Up(21, 18) # 18+18
        self.up7 = Up(18, 15) # 15
        self.lastAro = OutConv(15, self.num_class)
        self.lastSta = OutConvStage(15, 5)

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.conv7(x8)
        x = self.up0(x9, x8)
        x = self.up1(x, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        outputAro = self.lastAro(x)
        outputSta = self.lastSta(x)
        return outputAro, outputSta

class USleep(nn.Module):
    """
    Channel = 5
    """
    def __init__(self, size=4096*1024, channels=13, num_class=7):
        super(USleep, self).__init__()
        self.size = size
        self.channels = channels
        self.num_class = num_class
        self.conv01 = Down(self.channels, 15, 2)
        self.conv0 = Down(15, 18, 4) 
        self.conv1 = Down(18, 21, 4)
        self.conv2 = Down(21, 25, 4)
        self.conv3 = Down(25, 30, 4)
        self.conv4 = Down(30, 60, 4)
        self.conv5 = Down(60, 120, 4)
        self.conv6 = Down(120, 240, 4)
        self.conv7 = Down(240, 480, 4)
        self.up0 = Up(480, 240) # 240+240
        self.up1 = Up(240, 120) # 120+120
        self.up2 = Up(120, 60) # 60+60
        self.up3 = Up(60, 30) # 30+30
        self.up4 = Up(30, 25) # 25+25
        self.up5 = Up(25, 21) # 21+21
        self.up6 = Up(21, 18) # 18+18
        self.up7 = Up(18, 15) # 15
        self.last = OutConv(15, self.num_class)

    def forward(self, x):
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x5 = self.conv3(x4)
        x6 = self.conv4(x5)
        x7 = self.conv5(x6)
        x8 = self.conv6(x7)
        x9 = self.conv7(x8)
        x = self.up0(x9, x8)
        x = self.up1(x, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        output = self.last(x)
        return output

class Unet_test_sleep_data(nn.Module):
    """
    Channel = 8
    """
    def __init__(self, size=60*5*100, channels=5, num_class=1):
        super(Unet_test_sleep_data, self).__init__()
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
        self.Arousallast = OutConv(15, self.num_class)
        self.Apnealast = OutConv(15, self.num_class)

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
        Arousaloutput = self.Arousallast(x)
        Apneaoutput = self.Apnealast(x)
        return Arousaloutput, Apneaoutput

if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net = Unet_test_sleep_data(size=60*5*100, channels=8, num_class=1)
    net.to(DEVICE)
    summary(net, (8, 8, 5*60*100), device=DEVICE)
    
    
        
