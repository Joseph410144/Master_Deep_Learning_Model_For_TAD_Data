import numpy as np
import os
from tqdm import tqdm


label105_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\105\5chTest5min_del10min\Label"
label107_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\107\5chTest5min_del3min\Label"
label104_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\104\5chTest5min_del3min\Label"

datas = os.listdir(label104_datapath)
allPOints4 = 0
allArousals4 = 0
for data in tqdm(datas):
    Path = os.path.join(label104_datapath, data)
    lbl = np.load(Path)
    lbl = lbl.squeeze()
    allPOints4 += len(lbl)
    allArousals4 += np.sum(lbl==1)


datas = os.listdir(label105_datapath)
allPOints5 = 0
allArousals5 = 0
for data in tqdm(datas):
    Path = os.path.join(label105_datapath, data)
    lbl = np.load(Path)
    lbl = lbl.squeeze()
    allPOints5 += len(lbl)
    allArousals5 += np.sum(lbl==1)


datas = os.listdir(label107_datapath)
allPOints7 = 0
allArousals7 = 0
for data in tqdm(datas):
    Path = os.path.join(label107_datapath, data)
    lbl = np.load(Path)
    lbl = lbl.squeeze()
    allPOints7 += len(lbl)
    allArousals7 += np.sum(lbl==1)


print(f"104y RecordSec: {(allPOints4//100)}, ArousalSec: {(allArousals4//100)}")
print(f"105y RecordSec: {(allPOints5//100)}, ArousalSec: {(allArousals5//100)}")
print(f"107y RecordSec: {(allPOints7//100)}, ArousalSec: {(allArousals7//100)}")
