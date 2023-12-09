import os
from UnetModel import UsleepModLstm
import torch
from torch.nn.parallel import DataParallel
from Make107Valdata import MakeValData
import pandas as pd
from tqdm import tqdm
import argparse

def Accuracy(gt, predict, threshold):
    gt = gt.view(-1)
    predict = predict.view(-1)
    predict = torch.where(torch.ge(predict, threshold), 1, predict)
    predict = torch.where(torch.lt(predict, threshold), 0, predict)
    gt_pos_sum = (gt==1).sum().item()
    pre_pos_sum = (predict==1).sum().item()
    True_pos_sum = ((gt==1)*(predict==1)).sum().item()
    if pre_pos_sum==0:
        precision = 0
    else:
        precision = True_pos_sum/pre_pos_sum
    recall = True_pos_sum/gt_pos_sum
    return  precision, recall


def main(path, model, device):
    Val, Label, Epochs = MakeValData(path)
    if not Val.any():
        return -1, -1
    Precision_all = []
    Recall_all = []
    print("======== Validate Patient ========")
    for epoch in range(Epochs):
        ValT = Val[:, epoch*100*5*60:(epoch+1)*100*5*60]
        LabelT = Label[:, epoch*100*5*60:(epoch+1)*100*5*60]
        if 1 in LabelT[:, :]:
            X = torch.tensor(ValT).to(device=device, dtype=torch.float32)
            y = torch.tensor(LabelT).to(device=device, dtype=torch.float32)
            X = torch.reshape(X, (1, X.size(0), X.size(1)))
            pre = model(X)
            prec, reca = Accuracy(y, pre, 0.5)
            Precision_all.append(prec)
            Recall_all.append(reca)

    print("======== Validate Complete ========")
    if not Precision_all:
        return -1, -1
    return round(sum(Precision_all)/len(Precision_all), 2), round(sum(Recall_all)/len(Recall_all), 2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("Class")
    args = parser.parse_args()
    path = rf"D:\Joseph_NCHU\Lab\data\北醫UsleepData\AHIClass\only107\{args.Class}"
    Record = {"Name":[], "Precision":[], "Recall":[], "Note":[]}
    Patients = os.listdir(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UsleepModLstm.ULstmAutoencoder_5min(size=5*60*100, num_class=1, n_features=5)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)

    # Using Unet + BiLSTM
    model.load_state_dict(torch.load(r'weight\Apnea\Train_0929\model99_0.8375806574624548.pth'))
    model = model.eval()
    for patient in tqdm(Patients):
        path_patient = os.path.join(path, patient)
        precision, recall = main(path_patient, model, device)
        if precision != -1:
            Record["Name"].append(patient)
            Record["Precision"].append(precision)
            Record["Recall"].append(recall)
            Record["Note"].append(0)
        else:
            Record["Name"].append(patient)
            Record["Precision"].append(0)
            Record["Recall"].append(0)
            Record["Note"].append(1)
    
    df = pd.DataFrame(Record)
    df.to_csv(rf"D:\Joseph_NCHU\Lab\data\北醫UsleepData\AHIClass\only107\StaticOnTRaining\Static107{args.Class}.csv", encoding="utf_8_sig")