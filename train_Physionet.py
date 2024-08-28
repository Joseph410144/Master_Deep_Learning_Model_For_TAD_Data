import torch
import os
import math
import logging
import matplotlib.pyplot as plt

from Model import Loss_Function, TIEN_RisdualBiLSTM
from torch import optim
from SleepDataset import SleepDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DataParallel
from torchinfo import summary

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a", encoding='utf-8-sig')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def Accuracy(gt, predict, mode):
    gt = gt.view(-1)
    predict = predict.view(-1)
    predict = torch.round(predict)
    gt_pos_sum = (gt==1).sum().item()
    pre_pos_sum = (predict==1).sum().item()
    True_pos_sum = ((gt==1)*(predict==1)).sum().item()
    if pre_pos_sum==0:
        precision = 0
    else:
        precision = True_pos_sum/pre_pos_sum
    if gt_pos_sum==0:
        recall = -1
    else:
        recall = True_pos_sum/gt_pos_sum
    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall / (precision+recall)

    return f1_score, recall, precision

def DrawPicture(DictVar, WeightDataPath, traget):
    figures = [("f1_fig", "f1_test_fig"), ("re_fig", "re_test_fig"), ("pre_fig", "pre_test_fig"), ("loss_fig", "loss_test_fig")]
    for train, test in figures:
        plt.figure()
        plt.title(f"Train Val {traget} {train}")
        plt.xlabel("epoch")
        plt.ylabel(f"{train}")
        plt.plot(DictVar[train], label="train")
        plt.plot(DictVar[test], label="val")
        plt.legend()
        plt.savefig(rf"{WeightDataPath}\Figure\{traget}_{train}.jpg")
        plt.close()

def train(net, device, epochs, lr, train_loader, test_loader, logger, WeightDataPath):
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = Loss_Function.CrossEntropy_Cut_Physionet()
    best_loss = float('inf')
    bestTestLoss = float('inf')
    bestArousalF1 = 0

    """ record Training information """
    ArousalEpochRecord = {"loss_fig":[], "loss_test_fig":[], "f1_fig":[], "re_fig":[], "pre_fig":[], "f1_test_fig":[], 
                   "re_test_fig":[], "pre_test_fig":[]}

    
    Total_Loss = []
    Val_Loss = []
    epochs_list = range(epochs)
    if os.path.isfile(rf"{WeightDataPath}\CheckPoint.pt"):
        checkpoint = torch.load(rf"{WeightDataPath}\CheckPoint.pt")
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_ = checkpoint['epoch']
        best_loss = checkpoint['bestTrainLoss']
        bestTestLoss = checkpoint["bestTestLoss"]
        bestArousalF1 = checkpoint["bestArousalf1"]
        Total_Loss = checkpoint["Total_Loss"]
        Val_Loss = checkpoint["Val_Loss"]
        ArousalEpochRecord = checkpoint["ArousalEpochRecord"]
        epochs_list = epochs_list[epoch_:]

    """ Start Training """
    for epoch in epochs_list:
        ArousalStepRecord = {"f1ScoreAverage":0, "recallAverage":0, "precisionAverage":0, "lossAverage":0}
        Total_loss_Train = 0

        nums = 0
        ArousalNums = 0
        """ set Model into training mode """
        net.train()
        logger.info(f"********* Train Epoch:{epoch} ********")
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            ArousalLabel = label.to(device=device, dtype=torch.float32)
            ArousalPred = net(image)
            """ Calculate Loss """
            lossArousal, judgeArousal = criterion(ArousalPred, ArousalLabel)
            # 讓其中一個module不要倒傳遞達成只訓練一個module
            loss = lossArousal
            if math.isnan(loss):
                torch.save(net.state_dict(), rf'{WeightDataPath}\Loss_Nan_Train_model.pth')

            # judge, f1Score, recall, precision = Accuracy_Physionet(label, pred, "train")
            Arousalf1Score, Arousalrecall, Arousalprecision = Accuracy(ArousalLabel, ArousalPred, "train")
            if Arousalrecall != -1:
                ArousalStepRecord["f1ScoreAverage"] += Arousalf1Score
                ArousalStepRecord["recallAverage"] += Arousalrecall
                ArousalStepRecord["precisionAverage"] += Arousalprecision
                ArousalNums += 1
            
            """ Save model which has minimum loss"""
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), rf'{WeightDataPath}\best_Train_model.pth')

            """ Back propagation """
            # if judgeArousal and judgeApnea:
            #     loss.backward()
            # elif judgeArousal and (not judgeApnea):
            #     lossArousal.backward()
            # elif (not judgeArousal) and judgeApnea:
            #     lossApnea.backward()
            loss.backward()

            """ record loss """
            Total_loss_Train+=loss.item()
            ArousalStepRecord["lossAverage"] += lossArousal.item()
            nums += 1
            optimizer.step()
        
        ArousalStepRecord["f1ScoreAverage"] /= ArousalNums
        ArousalStepRecord["recallAverage"] /= ArousalNums
        ArousalStepRecord["precisionAverage"] /= ArousalNums
        ArousalStepRecord["lossAverage"] /= nums

        Total_loss_Train /= nums

        logger.info("\nTrain mode Epoch {}: Total loss:{} \nArousal >> F1_score:{}, recall:{}, precision{}, loss:{}".format(epoch, 
                    round(Total_loss_Train, 4),
                    round(ArousalStepRecord["f1ScoreAverage"], 4), round(ArousalStepRecord["recallAverage"], 4), round(ArousalStepRecord["precisionAverage"], 4), round(ArousalStepRecord["lossAverage"], 4)))
        
        ArousalEpochRecord["f1_fig"].append(ArousalStepRecord["f1ScoreAverage"])
        ArousalEpochRecord["re_fig"].append(ArousalStepRecord["recallAverage"])
        ArousalEpochRecord["pre_fig"].append(ArousalStepRecord["precisionAverage"])
        ArousalEpochRecord["loss_fig"].append(ArousalStepRecord["lossAverage"])
        
        Total_Loss.append(Total_loss_Train)
        logger.info("************************************\n")
        """ Validation Data """
        net.eval()
        ValArousal, ValLoss_, ArousalTestF1 = test(net, test_loader, criterion, device, logger)
        if ArousalTestF1 > bestArousalF1:
            bestArousalF1 = ArousalTestF1

        ArousalEpochRecord["f1_test_fig"].append(ValArousal["f1_score"])
        ArousalEpochRecord["re_test_fig"].append(ValArousal["recall"])
        ArousalEpochRecord["pre_test_fig"].append(ValArousal["precision"])
        ArousalEpochRecord["loss_test_fig"].append(ValArousal["loss"])

        Val_Loss.append(ValLoss_)

        if ValLoss_ < bestTestLoss:
            torch.save(net.state_dict(), rf'{WeightDataPath}\model{epoch}_{ValLoss_}.pth')
            bestTestLoss = ValLoss_
        
        torch.save({
            "epoch":epoch,
            "model":net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "bestTrainLoss": best_loss,
            "bestTestLoss": bestTestLoss,
            "bestArousalf1": bestArousalF1,
            "Total_Loss": Total_Loss,
            "Val_Loss": Val_Loss,
            "ArousalEpochRecord": ArousalEpochRecord,
        }, rf"{WeightDataPath}\CheckPoint.pt")

    DrawPicture(ArousalEpochRecord, WeightDataPath, "Arousal")

    plt.figure()
    plt.title(f"Total Loss Fig")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(Total_Loss, label="train")
    plt.plot(Val_Loss, label="val")
    plt.legend()
    plt.savefig(rf"{WeightDataPath}\Figure\Total_loss.jpg")
    plt.close()
    logger.info(f"Best Test Loss: {round(bestTestLoss, 4)}, Best Arousal F1 Score: {bestArousalF1}")

def test(net, test_iter, criterion, device, logger):
    TestArousalRecord = {"f1_score":0, "recall":0, "precision":0, "loss":0}
    total_loss_test = 0
    nums_ = 0
    ArousalNums_ = 0

    with torch.no_grad():
        logger.info("*************** test ***************")
        for X, y in tqdm(test_iter):
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            # t = t.to(device=device, dtype=torch.float32)
            # 只用5 channel 訓練
            X = X.contiguous()
            ArousalPred = net(X)
            ArousalLabel = y.contiguous()
            ArousalLoss, _ = criterion(ArousalPred, ArousalLabel)
            loss = ArousalLoss
            total_loss_test += loss.item()

            f1sc, RecallTest, PrecisionTest= Accuracy(ArousalLabel, ArousalPred, "test")
            if RecallTest != -1:
                TestArousalRecord["f1_score"] += f1sc
                TestArousalRecord["recall"] += RecallTest
                TestArousalRecord["precision"] += PrecisionTest
                ArousalNums_ += 1
            TestArousalRecord["loss"] += ArousalLoss.item()
            nums_ += 1
    
    TestArousalRecord["f1_score"] /= ArousalNums_
    TestArousalRecord["recall"] /= ArousalNums_
    TestArousalRecord["precision"] /= ArousalNums_
    TestArousalRecord["loss"] /= nums_
    total_loss_test /= nums_


    logger.info("\nTest mode: Total loss:{} \nArousal >> F1_score:{}, recall:{}, precision{}, loss:{}".format(round(total_loss_test, 4),
                    round(TestArousalRecord["f1_score"], 4), round(TestArousalRecord["recall"], 4), round(TestArousalRecord["precision"], 4), round(TestArousalRecord["loss"], 4)))
    logger.info("************************************\n")

    return TestArousalRecord, total_loss_test, round(TestArousalRecord["f1_score"], 4)


BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_CLASSES = 1
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
# NUM_PRINT = 11000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    WeightDataPath = r"weight\Physionet2018\Train_0806"
    """ Physionet2018 """
    TrainDatasetPath = r"E:\JosephHsiang\Physionet2018TrainData\Train"
    LabelDatasetPath = r"E:\JosephHsiang\Physionet2018TrainData\Label"
    allDataset = SleepDataset(rootX = TrainDatasetPath, rooty = LabelDatasetPath,
                      transform=None)
    train_size = int(len(allDataset)*0.8)
    test_size = len(allDataset)-train_size
    trainset, valset = torch.utils.data.random_split(allDataset, [train_size, test_size])
    
    trainloader = DataLoader(dataset=trainset,
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        drop_last=True)
    
    valloader = DataLoader(dataset=valset,
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        drop_last=True)


    logger = get_logger(f'{WeightDataPath}\TrainingNote.log')
    model_mod = TIEN_RisdualBiLSTM.ArousalApneaModel_Physionet(size=5*60*200, num_class=1, n_features=8)
    with open(f'{WeightDataPath}\Model.log', 'w', encoding='utf-8-sig') as f:
        report = summary(model_mod, input_size=(8, 8, 5*60*200), device=DEVICE)
        f.write(str(report))

    logger.info(f"Batch size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}, Loss: CrossEntropy, Optimizer: RMSProp, Device: {DEVICE}")
    logger.info(f"Phtsionet 2018 Arousal Test")
    model_mod.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model_mod = DataParallel(model_mod)
    train(model_mod, DEVICE, NUM_EPOCHS, LEARNING_RATE, trainloader, valloader, logger, WeightDataPath)


if __name__ == "__main__":
    main()