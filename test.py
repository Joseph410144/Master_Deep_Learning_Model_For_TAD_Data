import numpy as np
import logging
import torch
import torch.nn.functional as F
from torchvision.models import resnet34
from torchinfo import summary

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def SignalSegAndOvp(series):
    pad = torch.nn.ZeroPad2d(padding=(25, 25, 0, 0))
    overlapped_series = torch.zeros(8, 200, 200)
    series_test = pad(series)
    for i in range(0, 200):
        overlapped_series[:, i, :] = series_test[:, i*150:(i*150)+200]
    return overlapped_series

def SignalReduction(series):
    original_series = torch.zeros(8, 200, 150)
    for i in range(0, 200):
        original_series[:, i, :] = series[:, i, 25:175]

    original_series = original_series.view(8, 200*150).contiguous()
    return original_series

def pos_encoding(t, Time_Dimension):
    pos_enc_all = torch.zeros(t.shape[0], 128)
    for index in range(0, t.shape[0]):
        t_emb = t[index][0]
        t_emb = torch.tensor([t_emb])
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, Time_Dimension, 2).float() / Time_Dimension)
        )
        pos_enc_a = torch.sin(t_emb.repeat(1, Time_Dimension // 2) * inv_freq)
        pos_enc_b = torch.cos(t_emb.repeat(1, Time_Dimension // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        pos_enc_all[index, :] = pos_enc[0, :]
    return pos_enc_all

ss = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
print(f"input shape: {ss.shape}")
position = pos_encoding(ss, 128) # time step = 1, embedding dimension = 128
lin = torch.nn.Linear(128, 8)
ans = lin(position)[:, :, None].repeat(1, 1, 30000) 
print(ans.shape)
print(f"Embedding Dimension: {position.shape}")
# net = resnet34()
# report = summary(net, input_size=(1, 3, 150, 120), device="cpu")