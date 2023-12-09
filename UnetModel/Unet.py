import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchsummary import summary


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

        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=4, padding=0)
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
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
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

# class USleep(nn.Module):
#     """
#     Channel = 13
#     """
#     def __init__(self, size=4096*2048, channels=13, num_class=7):
#         super(USleep, self).__init__()
#         self.size = size
#         self.channels = channels
#         self.num_class = num_class
#         self.conv01 = Down(self.channels, 20, 2)
#         self.conv0 = Down(20, 40, 4) 
#         self.conv1 = Down(40, 48, 4)
#         self.conv2 = Down(48, 64, 4)
#         self.conv3 = Down(64, 80, 4)
#         self.conv4 = Down(80, 112, 4)
#         self.conv5 = Down(112, 144, 4)
#         self.conv6 = Down(144, 208, 4)
#         self.conv7 = Down(208, 272, 4)
#         self.conv8 = Down(272, 400, 4)
#         self.conv9 = Down(400, 528, 4)
#         self.conv10 = Down(528, 1024, 4)
#         self.up0 = Up(1024, 528) # 240+240
#         self.up1 = Up(528, 400) # 120+120
#         self.up2 = Up(400, 272) # 60+60
#         self.up3 = Up(272, 208) # 30+30
#         self.up4 = Up(208, 144) # 25+25
#         self.up5 = Up(144, 112) # 21+21
#         self.up6 = Up(112, 80) # 18+18
#         self.up7 = Up(80, 64) # 15
#         self.up8 = Up(64, 48) # 15
#         self.up9 = Up(48, 40) # 15
#         self.up10 = Up(40, 20) # 15
#         self.last = OutConv(20, self.num_class)

#     def forward(self, x):
#         x1 = self.conv01(x)
#         x2 = self.conv0(x1)
#         x3 = self.conv1(x2)
#         x4 = self.conv2(x3)
#         x5 = self.conv3(x4)
#         x6 = self.conv4(x5)
#         x7 = self.conv5(x6)
#         x8 = self.conv6(x7)
#         x9 = self.conv7(x8)
#         x10 = self.conv8(x9)
#         x11 = self.conv9(x10)
#         x12 = self.conv10(x11)
#         x = self.up0(x12, x11)
#         x = self.up1(x, x10)
#         x = self.up2(x, x9)
#         x = self.up3(x, x8)
#         x = self.up4(x, x7)
#         x = self.up5(x, x6)
#         x = self.up6(x, x5)
#         x = self.up7(x, x4)
#         x = self.up8(x, x3)
#         x = self.up9(x, x2)
#         x = self.up10(x, x1)
#         output = self.last(x)
#         return output

if __name__ == '__main__':
    DEVICE = "cpu"
    net = USleepStage(size=4096*2048, channels=13, num_class=1)
    net.to(DEVICE)
    summary(net, (13, 4096*2048), device=DEVICE)
    
    
        
