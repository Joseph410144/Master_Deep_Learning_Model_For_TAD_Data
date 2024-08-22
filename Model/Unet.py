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
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )


    def forward(self, x, x_o):
        x = self.up(x)
        # x = torch.cat((x, x_o), 1)
        return self.conv(x)

class Unet_SingleTask(nn.Module):
    """
    Channel = 8
    """
    def __init__(self, size=60*5*100, channels=8, num_class=1):
        super(Unet_SingleTask, self).__init__()
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

        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.channels, self.num_class, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_o = x
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x, x_o)
        output = self.output(x)
        return output

class Unet_MultiTask(nn.Module):
    """
    Channel = 8
    """
    def __init__(self, size=60*5*100, channels=8, num_class=1):
        super(Unet_MultiTask, self).__init__()
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

        self.outputArousal = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.channels, self.num_class, kernel_size=1),
            nn.Sigmoid()
        )

        self.outputApnea = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.channels, self.num_class, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_o = x
        x1 = self.conv01(x)
        x2 = self.conv0(x1)
        x3 = self.conv1(x2)
        x4 = self.conv2(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.last(x, x_o)
        outputArousal = self.outputArousal(x)
        outputApnea = self.outputApnea(x)

        return outputArousal, outputApnea

class Unet(nn.Module):
    """
    Channel = 8
    """
    def __init__(self, size=60*5*100, channels=8, num_class=1):
        super(Unet, self).__init__()
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

        # self.ouput = 

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
        
