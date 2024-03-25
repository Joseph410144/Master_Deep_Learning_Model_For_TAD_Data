import torch
import torch.nn as nn
import torch.nn.functional as F
### torchsummary don't support RNN
# from torchsummary import summary
from torchinfo import summary
from torch.nn.parallel import DataParallel

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
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x

        return res
    
class TimesNet(nn.Module):
    def __init__(self, seq_length, num_class, n_features, layer):
        super(TimesNet, self).__init__()
        self.seq_length = seq_length
        self.num_class = num_class
        self.n_features = n_features
        self.layer = layer

        self.layer_norm = nn.LayerNorm(8)
        self.Timesblock = nn.ModuleList([TimesBlock(seq_len=self.seq_length, pred_len=0, top_k=5, d_model=8, d_ff=32, 
                                                    num_kernels=3, num_class=self.num_class, n_features=self.n_features)
                                    for _ in range(self.layer)])
        self.Arousalclassifier = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.n_features, self.num_class, kernel_size=1),
            nn.Sigmoid()
        )

        self.Apneaclassifier = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.n_features, self.num_class, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_out = x
        for i in range(self.layer):
            x_out = self.layer_norm(self.Timesblock[i](x_out))
            x_out = x_out.permute(0, 2, 1)

        x_Arousal = self.Arousalclassifier(x_out)
        x_Apnea = self.Apneaclassifier(x_out)
        
        return x_Arousal, x_Apnea, x_out
    

if __name__ == "__main__":
    net = TimesNet(seq_length=5*60*100, num_class=1, n_features=8, layer=3)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(DEVICE)
    report = summary(net, input_size=(8, 8, 5*60*100), device=DEVICE)