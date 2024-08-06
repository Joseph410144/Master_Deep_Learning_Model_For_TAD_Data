import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from DatasetUnet import UnetDataset
from torcheval.metrics.functional import binary_auprc, binary_auroc
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torch.utils.data import DataLoader
from Model import TIEN_RisdualBiLSTM
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
    for threshold in np.arange(0, 1, 0.01):
        totalArousalPre = 0
        totalArousalRec = 0
        num = 0
        for X, y in tqdm(test_iter):
            num+=1
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            ArousalLabel = y.contiguous()
            Arousalpred= model(X)
            Arousalprec, Arousalreca = Accuracy(ArousalLabel, Arousalpred, threshold)

            totalArousalPre += Arousalprec
            totalArousalRec += Arousalreca

        precisionAllArousal.append(totalArousalPre/num)
        recallAllArousal.append(totalArousalRec/num)

    plt.plot(recallAllArousal, precisionAllArousal)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(rf"weight\Arousal_Apnea\Train_0404\Arousalre_fig.jpg")
    print("AUPRC:", auc(recallAllArousal, precisionAllArousal))




def main():
    Valtrain_datapath = r"E:\JosephHsiang\Physionet2018TrainData\Test\Data"
    Vallabel_datapath = r"E:\JosephHsiang\Physionet2018TrainData\Test\Label"


    logger = get_logger(fr'weight\Physionet2018\Train_0711\TestingNote.log')
    logger.info(f"Using Physionet 2018 Data for testing")
    # logger.info(f"Change linear to Coonv1D 1x1 to reduct dim")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = TimesUnet.TimesUnet(size=60*5*100, channels=8, num_class=1)
    model = TIEN_RisdualBiLSTM.ArousalApneaModel_Physionet(size=5*60*200, num_class=1, n_features=8)
    # model = TimesNet.TimesNet(seq_length=5*60*100, num_class=1, n_features=8, layer=3)
    # model = Unet.Unet_test_sleep_data(size=60*5*100, channels=8, num_class=1)
    # model = DPRNNBlock.DPRNNClassifier(size=5*60*100, num_class=1, n_features=8)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.load_state_dict(torch.load(rf'weight\Physionet2018\Train_0711\model44_1.048349263717991.pth'))
    model = model.eval()
    allDataset = UnetDataset(rootX = Valtrain_datapath, rooty = Vallabel_datapath,
                      transform=None)
    test_iter = DataLoader(dataset=allDataset,
                        batch_size=1, 
                        drop_last=True,
                        shuffle=False)

    
    totalArousalPre = 0
    totalArousalRec = 0
    ArousalNums = 0
    ArousalAUPRC = 0
    ArousalAUROC = 0
    thresholds = torch.linspace(0.01, 0.9, steps=50)
    metricArousal = BinaryPrecisionRecallCurve(thresholds=thresholds)

    for X, y in tqdm(test_iter):
        X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
        ArousalPred = model(X)
        ArousalLabel = y.contiguous()
        if CheckDataHasAnomaly(ArousalLabel):
            ArousalNums += 1
            prec, reca = Accuracy(ArousalLabel, ArousalPred, 0.5)
            totalArousalPre += prec
            totalArousalRec += reca
            ArousalPred = ArousalPred.squeeze()
            ArousalLabel = ArousalLabel.squeeze()
            """ remove -1 label """
            mask = ArousalLabel != -1
            ArousalLabel = ArousalLabel[mask]
            ArousalPred = ArousalPred[mask]
            """ Calculate AUPRC, AUROC """
            ArousalAUPRC += binary_auprc(ArousalPred, ArousalLabel).item()
            ArousalAUROC += binary_auroc(ArousalPred, ArousalLabel).item()
            # metricArousal.update(ArousalPred[:, 0, :].squeeze().cpu(), ArousalLabel.squeeze().to(torch.long).cpu())

    AverageArousalPre = round(totalArousalPre/ArousalNums, 3)
    AverageArousalRec = round(totalArousalRec/ArousalNums, 3)
    AverageArousalF1Score = round(2*AverageArousalPre*AverageArousalRec/(AverageArousalPre+AverageArousalRec), 2)
    logger.info("Arousal Results")
    logger.info(fr"Average Arousal Precision:{AverageArousalPre}, Average Arousal Recall:{AverageArousalRec}, Average Arousal F1 Score:{AverageArousalF1Score}")
    logger.info(f"ArousalAUPRC: {ArousalAUPRC/ArousalNums}")
    logger.info(f"ArousalAUROC: {ArousalAUROC/ArousalNums}")    

    # fig_Arousal, ax_Arousal = metricArousal.plot(score=True)
    # ax_Arousal.set_title('Arousal Precision-Recall Curve')
    # plt.show()


    

if __name__ == "__main__":
    main()