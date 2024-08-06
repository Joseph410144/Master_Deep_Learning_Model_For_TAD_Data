import torch
import os
import logging
import matplotlib.pyplot as plt
import numpy as np

from DatasetUnet import UnetDataset
from torch.utils.data import DataLoader
from Model import USleepMod, UsleepModLstm, UnetLSTMModel, Unet, DPRNNBlock, TimesNet, TimesUnet, TIEN_RisdualBiLSTM
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from sklearn.metrics import auc

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def CheckDataHasAnomaly(LabelData):
    LabelData_f = LabelData.view(-1)
    mask = LabelData_f >= 0.5
    if not mask.any():
        judge = False
    else:
        judge = True

    return judge


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


def AUPRC(test_iter, model, device):
    precisionAllArousal = []
    recallAllArousal = []
    precisionAllApnea = []
    recallAllApnea = []
    for threshold in np.arange(0, 1, 0.01):
        totalArousalPre = 0
        totalArousalRec = 0
        totalApneaPre = 0
        totalApneaRec = 0
        num = 0
        for X, y in tqdm(test_iter):
            num+=1
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            ArousalLabel = y[:, 0, :].contiguous()
            ApneaLabel = y[:, 1, :].contiguous()
            Arousalpred, Apneapred = model(X)
            Arousalprec, Arousalreca = Accuracy(ArousalLabel, Arousalpred, threshold)
            Apneaprec, Apneareca = Accuracy(ApneaLabel, Apneapred, threshold)

            totalArousalPre += Arousalprec
            totalArousalRec += Arousalreca

            totalApneaPre += Apneaprec
            totalApneaRec += Apneareca

        precisionAllArousal.append(totalArousalPre/num)
        recallAllArousal.append(totalArousalRec/num)
        precisionAllApnea.append(totalApneaPre/num)
        recallAllApnea.append(totalApneaRec/num)

    plt.plot(recallAllArousal, precisionAllArousal)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(rf"weight\Arousal_Apnea\Train_0404\Arousalre_fig.jpg")
    print("AUPRC:", auc(recallAllArousal, precisionAllArousal))

    plt.plot(recallAllApnea, precisionAllApnea)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(rf"weight\Arousal_Apnea\Train_0404\Apneare_fig.jpg")
    print("AUPRC:", auc(recallAllApnea, precisionAllApnea))



def main():
    Valtrain_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Train"
    Vallabel_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Label"
    logger = get_logger(fr'weight\Arousal_Apnea\Train_0501\TestingNote.log')
    logger.info(f"Using TMU 107 Data for testing")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TIEN_RisdualBiLSTM.ArousalApneaModel(size=5*60*100, num_class=1, n_features=8)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.load_state_dict(torch.load(rf'weight\Arousal_Apnea\Train_0501\model48_1.788333569254194.pth'))
    model = model.eval()
    allDataset = UnetDataset(rootX = Valtrain_datapath, rooty = Vallabel_datapath,
                      transform=None)
    test_iter = DataLoader(dataset=allDataset,
                        batch_size=4, 
                        drop_last=True,
                        shuffle=False)

    
    totalArousalPre = 0
    totalArousalRec = 0
    totalApneaPre = 0
    totalApneaRec = 0
    ArousalNums = 0
    ApneaNums = 0
    for X, y in tqdm(test_iter):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        ArousalPred, ApneaPred = model(X)
        ArousalLabel = y[:, 0, :].contiguous()
        ApneaLabel = y[:, 1, :].contiguous()
        if CheckDataHasAnomaly(ArousalLabel):
            ArousalNums += 1
            prec, reca = Accuracy(ArousalLabel, ArousalPred, 0.5)
            totalArousalPre += prec
            totalArousalRec += reca
        
        if CheckDataHasAnomaly(ApneaLabel):
            ApneaNums += 1
            prec, reca = Accuracy(ApneaLabel, ApneaPred, 0.5)
            totalApneaPre += prec
            totalApneaRec += reca

    AverageArousalPre = round(totalArousalPre/ArousalNums, 3)
    AverageArousalRec = round(totalArousalRec/ArousalNums, 3)
    AverageArousalF1Score = round(2*AverageArousalPre*AverageArousalRec/(AverageArousalPre+AverageArousalRec), 2)
    logger.info("Arousal Results")
    logger.info(fr"Average Arousal Precision:{AverageArousalPre}, Average Arousal Recall:{AverageArousalRec}, Average Arousal F1 Score:{AverageArousalF1Score}")
    
    AverageApneaPre = round(totalApneaPre/ApneaNums, 3)
    AverageApneaRec = round(totalApneaRec/ApneaNums, 3)
    AverageApneaF1Score = round(2*AverageApneaPre*AverageApneaRec/(AverageApneaPre+AverageApneaRec), 2)
    logger.info("Apnea Results")
    logger.info(fr"Average Apnea Precision:{AverageApneaPre}, Average Apnea Recall:{AverageApneaRec}, Average Apnea F1 Score:{AverageApneaF1Score}")


    

if __name__ == "__main__":
    main()