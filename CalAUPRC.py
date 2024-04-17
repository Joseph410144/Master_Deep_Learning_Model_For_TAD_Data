from UnetModel import USleepMod, UsleepModLstm, UnetLSTMModel, Unet, DPRNNBlock, TimesNet, TimesUnet
import torch
from torchinfo import summary
from torcheval.metrics.functional import binary_auprc, binary_auroc
from torchmetrics.classification import BinaryPrecisionRecallCurve
from DatasetUnet import UnetDataset, UnetDataset_timeEmbd
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
import os
import logging
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


def AUPRC(test_iter, model, device, logger):
    precisionAllArousal = []
    recallAllArousal = []
    precisionAllApnea = []
    recallAllApnea = []
    for threshold in np.arange(0, 1.02, 0.01):
        totalArousalPre = 0
        totalArousalRec = 0
        totalApneaPre = 0
        totalApneaRec = 0
        ArousalNums = 0
        ApneaNums = 0
        
        for X, y in tqdm(test_iter):
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            ArousalLabel = y[:, 0, :].contiguous()
            ApneaLabel = y[:, 1, :].contiguous()
            Arousalpred, Apneapred = model(X)
            if CheckDataHasAnomaly(ArousalLabel):
                ArousalNums += 1
                Arousalprec, Arousalreca = Accuracy(ArousalLabel, Arousalpred, threshold)
                totalArousalPre += Arousalprec
                totalArousalRec += Arousalreca
            
            if CheckDataHasAnomaly(ApneaLabel):
                ApneaNums += 1
                Apneaprec, Apneareca = Accuracy(ApneaLabel, Apneapred, threshold)
                totalApneaPre += Apneaprec
                totalApneaRec += Apneareca


        precisionAllArousal.append(totalArousalPre/ArousalNums)
        recallAllArousal.append(totalArousalRec/ArousalNums)
        precisionAllApnea.append(totalApneaPre/ApneaNums)
        recallAllApnea.append(totalApneaRec/ApneaNums)

    plt.plot(recallAllArousal, precisionAllArousal)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(rf"weight\Arousal_Apnea\Train_0310\Arousalre_fig.jpg")
    plt.close()
    logger.info(f"AUPRC: {auc(recallAllArousal, precisionAllArousal)}")

    plt.plot(recallAllApnea, precisionAllApnea)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(rf"weight\Arousal_Apnea\Train_0310\Apneare_fig.jpg")
    logger.info(f"AUPRC: {auc(recallAllApnea, precisionAllApnea)}")



def main():
    Valtrain_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Train"
    Vallabel_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Label"


    logger = get_logger(fr'weight\Arousal_Apnea\Train_0310\TestingNote.log')
    logger.info(f"Using TMU 107 Data for calculating AUPRC")
    # logger.info(f"Change linear to Coonv1D 1x1 to reduct dim")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = TimesUnet.TimesUnet(size=60*5*100, channels=8, num_class=1)
    model = UnetLSTMModel.ArousalApneaUENNModel(size=5*60*100, num_class=1, n_features=8)
    # model = TimesNet.TimesNet(seq_length=5*60*100, num_class=1, n_features=8, layer=3)
    # model = Unet.Unet_test_sleep_data(size=60*5*100, channels=8, num_class=1)
    # model = DPRNNBlock.DPRNNClassifier(size=5*60*100, num_class=1, n_features=8)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.load_state_dict(torch.load(rf'weight\Arousal_Apnea\Train_0310\model42_1.061769385063915.pth'))
    model = model.eval()
    allDataset = UnetDataset(rootX = Valtrain_datapath, rooty = Vallabel_datapath,
                      transform=None)
    test_iter = DataLoader(dataset=allDataset,
                        batch_size=1, 
                        drop_last=True,
                        shuffle=False)
    
    thresholds = torch.linspace(0.01, 0.9, steps=50)

    ArousalNums = 0
    ArousalAUPRC = 0
    ArousalAUROC = 0
    metricArousal = BinaryPrecisionRecallCurve(thresholds=thresholds)

    ApneaNums = 0
    ApneaAUPRC = 0
    ApneaAUROC = 0
    metricApnea = BinaryPrecisionRecallCurve(thresholds=thresholds)

    for X, y in tqdm(test_iter):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        ArousalLabel = y[:, 0, :].contiguous()
        ApneaLabel = y[:, 1, :].contiguous()
        Arousalpred, Apneapred = model(X)
        if CheckDataHasAnomaly(ArousalLabel):
            ArousalNums += 1
            ArousalAUPRC += binary_auprc(Arousalpred[:, 0, :].squeeze(), ArousalLabel.squeeze()).item()
            ArousalAUROC += binary_auroc(Arousalpred[:, 0, :].squeeze(), ArousalLabel.squeeze()).item()
            metricArousal.update(Arousalpred[:, 0, :].squeeze().cpu(), ArousalLabel.squeeze().to(torch.long).cpu())
            # precision, recall, thresholds = metricArousal.compute()
            # print(len(precision), recall, thresholds, type(precision))
            # fig_, ax_ = metricArousal.plot(score=True)
            # plt.show()
            # plt.close()
            # metricArousal.reset()
            
         
        if CheckDataHasAnomaly(ApneaLabel):
            ApneaNums += 1
            ApneaAUPRC += binary_auprc(Apneapred[:, 0, :].squeeze(), ApneaLabel.squeeze()).item()
            ApneaAUROC += binary_auroc(Apneapred[:, 0, :].squeeze(), ApneaLabel.squeeze()).item()
            metricApnea.update(Apneapred[:, 0, :].squeeze().cpu(), ApneaLabel.squeeze().to(torch.long).cpu())
            # fig_, ax_ = metricApnea.plot(score=True)
            # plt.show()
            # plt.close()
            # metricApnea.reset()

    logger.info(f"ArousalAUPRC: {ArousalAUPRC/ArousalNums}")
    logger.info(f"ApneaAUPRC: {ApneaAUPRC/ApneaNums}")

    logger.info(f"ArousalAUROC: {ArousalAUROC/ArousalNums}")
    logger.info(f"ApneaAUROC: {ApneaAUROC/ApneaNums}")
    
    
    fig_Arousal, ax_Arousal = metricArousal.plot(score=True)
    ax_Arousal.set_title('Arousal Precision-Recall Curve')

    fig_Apnea, ax_Apnea = metricApnea.plot(score=True)
    ax_Apnea.set_title('Apnea Precision-Recall Curve')

    plt.show()


    # AUPRC(test_iter, model, device, logger)

if __name__ == "__main__":
    main()