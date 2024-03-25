from UnetModel import USleepMod, UsleepModLstm, UnetLSTMModel, Unet, DPRNNBlock, TimesNet
import torch
from torchinfo import summary
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


def AUPRC(test_iter, model, device):
    precisionAll = []
    recallAll = []
    for threshold in np.arange(0, 1, 0.01):
        totalPre = 0
        totalRec = 0
        num = 0
        for X, y in tqdm(test_iter):
            num+=1
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            ArousalLabel = y[:, 0, :].contiguous()
            ApneaLabel = y[:, 1, :].contiguous()
            pre = model(X)
            prec, reca = Accuracy(y, pre, threshold)
            totalPre += prec
            totalRec += reca
        precisionAll.append(totalPre/num)
        recallAll.append(totalRec/num)
    plt.plot(recallAll, precisionAll)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("P-R Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(rf"weight\Apnea\Train_0927\re_fig.jpg")
    print("AUPRC:", auc(recallAll, precisionAll))

def Point2Second(predict, gt, DEVICE):
    """
    input: Predict data and label(by point)
    output: Predict data(by second) and label(by second)
    """

    batch, gt_length = gt.shape
    gt = gt.view(batch, gt_length//100, 100).contiguous().to(DEVICE)
    gt_matrix_second = torch.ones(100, dtype=gt.dtype).view(100, 1).contiguous().to(DEVICE)
    gt_bySecond = torch.matmul(gt, gt_matrix_second)/100
    batch, length, _ = gt_bySecond.shape
    gt_bySecond = gt_bySecond.view(batch, length).contiguous()

    predict = predict.view(batch, gt_length//100, 100).contiguous().to(DEVICE)
    predict_bySecond = torch.matmul(predict, gt_matrix_second)/100
    batch, length, _ = predict_bySecond.shape
    predict_bySecond = predict_bySecond.view(batch, length).contiguous()

    return predict_bySecond, gt_bySecond

def main():
    Valtrain_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Train"
    Vallabel_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Validation\Label"


    logger = get_logger(fr'weight\Arousal_Apnea\Train_0126\TestingNote.log')
    logger.info(f"Using TMU 107 Data for testing")
    # logger.info(f"Change linear to Coonv1D 1x1 to reduct dim")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnetLSTMModel.ArousalApneaUENNModel(size=5*60*100, num_class=1, n_features=8)
    # model = TimesNet.TimesNet(seq_length=5*60*100, num_class=1, n_features=8, layer=3)
    # model = Unet.Unet_test_sleep_data(size=60*5*100, channels=8, num_class=1)
    # model = DPRNNBlock.DPRNNClassifier(size=5*60*100, num_class=1, n_features=8)
    # with open(fr'weight\Arousal_Apnea\Train_1108\Model.log', 'w', encoding='utf-8-sig') as f:
    #     report = summary(model, input_size=(8, 8, 5*60*100), device=device)
    #     f.write(str(report))
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.load_state_dict(torch.load(rf'weight\Arousal_Apnea\Train_0126\model38_1.133586593369814.pth'))
    model = model.eval()
    allDataset = UnetDataset(rootX = Valtrain_datapath, rooty = Vallabel_datapath,
                      transform=None)
    test_iter = DataLoader(dataset=allDataset,
                        batch_size=8, 
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
        # t = t.to(device=device, dtype=torch.float32)
        ArousalPred, ApneaPred = model(X)
        ArousalLabel = y[:, 0, :].contiguous()
        ApneaLabel = y[:, 1, :].contiguous()
        if CheckDataHasAnomaly(ArousalLabel):
            ArousalNums += 1
            pre_second, lab_second = Point2Second(ArousalPred, ArousalLabel, device)
            prec, reca = Accuracy(lab_second, pre_second, 0.5)
            totalArousalPre += prec
            totalArousalRec += reca
        
        if CheckDataHasAnomaly(ApneaLabel):
            ApneaNums += 1
            pre_second, lab_second = Point2Second(ApneaPred, ApneaLabel, device)
            prec, reca = Accuracy(lab_second, pre_second, 0.5)
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