from UnetModel import Loss_Function, UsleepModLstm, UnetLSTMModel
from torch import optim
import torch
from DatasetUnet import UnetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
import os
from torch.utils.data import ConcatDataset
import math
from torchinfo import summary
import logging



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
    criterion = Loss_Function.CrossEntropy_Cut()
    best_loss = float('inf')
    bestTestLoss = float('inf')
    bestArousalF1 = 0
    bestApneaF1 = 0

    """ record Training information """
    ArousalEpochRecord = {"loss_fig":[], "loss_test_fig":[], "f1_fig":[], "re_fig":[], "pre_fig":[], "f1_test_fig":[], 
                   "re_test_fig":[], "pre_test_fig":[]}

    ApneaEpochRecord = {"loss_fig":[], "loss_test_fig":[], "f1_fig":[], "re_fig":[], "pre_fig":[], "f1_test_fig":[], 
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
        bestApneaF1 = checkpoint["bestApneaf1"]
        Total_Loss = checkpoint["Total_Loss"]
        Val_Loss = checkpoint["Val_Loss"]
        ArousalEpochRecord = checkpoint["ArousalEpochRecord"]
        ApneaEpochRecord = checkpoint["ApneaEpochRecord"]
        epochs_list = epochs_list[epoch_:]

    """ Start Training """
    for epoch in epochs_list:
        ArousalStepRecord = {"f1ScoreAverage":0, "recallAverage":0, "precisionAverage":0, "lossAverage":0}
        ApneaStepRecord = {"f1ScoreAverage":0, "recallAverage":0, "precisionAverage":0, "lossAverage":0}
        Total_loss_Train = 0

        nums = 0
        ArousalNums = 0
        ApneaNums = 0
        """ set Model into training mode """
        net.train()
        logger.info(f"********* Train Epoch:{epoch} ********")
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            ArousalLabel = label[:, 0, :].contiguous()
            ApneaLabel = label[:, 1, :].contiguous()
            ArousalPred, ApneaPred = net(image)       
            """ Calculate Loss """
            lossArousal, judgeArousal = criterion(ArousalPred, ArousalLabel)
            lossApnea, judgeApnea = criterion(ApneaPred, ApneaLabel)
            loss = lossApnea+lossArousal
            if math.isnan(loss):
                torch.save(net.state_dict(), rf'{WeightDataPath}\Loss_Nan_Train_model.pth')

            # judge, f1Score, recall, precision = Accuracy_Physionet(label, pred, "train")
            Arousalf1Score, Arousalrecall, Arousalprecision = Accuracy(ArousalLabel, ArousalPred, "train")
            if Arousalrecall != -1:
                ArousalStepRecord["f1ScoreAverage"] += Arousalf1Score
                ArousalStepRecord["recallAverage"] += Arousalrecall
                ArousalStepRecord["precisionAverage"] += Arousalprecision
                ArousalNums += 1

            Apneaf1Score, Apnearecall, Apneaprecision = Accuracy(ApneaLabel, ApneaPred, "train")
            if Apnearecall != -1:
                ApneaStepRecord["f1ScoreAverage"] += Apneaf1Score
                ApneaStepRecord["recallAverage"] += Apnearecall
                ApneaStepRecord["precisionAverage"] += Apneaprecision
                ApneaNums += 1
            
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
            ApneaStepRecord["lossAverage"] += lossApnea.item()
            nums += 1
            optimizer.step()
        
        ArousalStepRecord["f1ScoreAverage"] /= ArousalNums
        ArousalStepRecord["recallAverage"] /= ArousalNums
        ArousalStepRecord["precisionAverage"] /= ArousalNums
        ArousalStepRecord["lossAverage"] /= nums

        ApneaStepRecord["f1ScoreAverage"] /= ApneaNums
        ApneaStepRecord["recallAverage"] /= ApneaNums
        ApneaStepRecord["precisionAverage"] /= ApneaNums
        ApneaStepRecord["lossAverage"] /= nums

        Total_loss_Train /= nums

        logger.info("\nTrain mode Epoch {}: Total loss:{} \nArousal >> F1_score:{}, recall:{}, precision{}, loss:{} \nApnea >> F1_score:{}, recall:{}, precision{}, loss:{}".format(epoch, 
                    round(Total_loss_Train, 2),
                    round(ArousalStepRecord["f1ScoreAverage"], 2), round(ArousalStepRecord["recallAverage"], 2), round(ArousalStepRecord["precisionAverage"], 2), round(ArousalStepRecord["lossAverage"], 2),
                    round(ApneaStepRecord["f1ScoreAverage"], 2), round(ApneaStepRecord["recallAverage"], 2), round(ApneaStepRecord["precisionAverage"], 2), round(ApneaStepRecord["lossAverage"], 2)))
        
        ArousalEpochRecord["f1_fig"].append(ArousalStepRecord["f1ScoreAverage"])
        ArousalEpochRecord["re_fig"].append(ArousalStepRecord["recallAverage"])
        ArousalEpochRecord["pre_fig"].append(ArousalStepRecord["precisionAverage"])
        ArousalEpochRecord["loss_fig"].append(ArousalStepRecord["lossAverage"])

        ApneaEpochRecord["f1_fig"].append(ApneaStepRecord["f1ScoreAverage"])
        ApneaEpochRecord["re_fig"].append(ApneaStepRecord["recallAverage"])
        ApneaEpochRecord["pre_fig"].append(ApneaStepRecord["precisionAverage"])
        ApneaEpochRecord["loss_fig"].append(ApneaStepRecord["lossAverage"])
        
        Total_Loss.append(Total_loss_Train)
        logger.info("************************************\n")
        """ Validation Data """
        net.eval()
        ValArousal, ValApnea, ValLoss_, ArousalTestF1, ApneaTestF1 = test(net, test_loader, criterion, device, logger)
        if ArousalTestF1 > bestArousalF1:
            bestArousalF1 = ArousalTestF1
        if ApneaTestF1 > bestApneaF1:
            bestApneaF1 = ApneaTestF1

        ArousalEpochRecord["f1_test_fig"].append(ValArousal["f1_score"])
        ArousalEpochRecord["re_test_fig"].append(ValArousal["recall"])
        ArousalEpochRecord["pre_test_fig"].append(ValArousal["precision"])
        ArousalEpochRecord["loss_test_fig"].append(ValArousal["loss"])

        ApneaEpochRecord["f1_test_fig"].append(ValApnea["f1_score"])
        ApneaEpochRecord["re_test_fig"].append(ValApnea["recall"])
        ApneaEpochRecord["pre_test_fig"].append(ValApnea["precision"])
        ApneaEpochRecord["loss_test_fig"].append(ValApnea["loss"])

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
            "bestApneaf1": bestApneaF1,
            "Total_Loss": Total_Loss,
            "Val_Loss": Val_Loss,
            "ArousalEpochRecord": ArousalEpochRecord,
            "ApneaEpochRecord": ApneaEpochRecord
        }, rf"{WeightDataPath}\CheckPoint.pt")

    DrawPicture(ArousalEpochRecord, WeightDataPath, "Arousal")
    DrawPicture(ApneaEpochRecord, WeightDataPath, "Apnea")

    plt.figure()
    plt.title(f"Total Loss Fig")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(Total_Loss, label="train")
    plt.plot(Val_Loss, label="val")
    plt.legend()
    plt.savefig(rf"{WeightDataPath}\Figure\Total_loss.jpg")
    plt.close()
    logger.info(f"Best Test Loss: {round(bestTestLoss, 2)}, Best Arousal F1 Score: {bestArousalF1}, Best Apnea F1 Score: {bestApneaF1}")

def test(net, test_iter, criterion, device, logger):
    TestArousalRecord = {"f1_score":0, "recall":0, "precision":0, "loss":0}
    TestApneaRecord = {"f1_score":0, "recall":0, "precision":0, "loss":0}
    total_loss_test = 0
    nums_ = 0
    ArousalNums_ = 0
    ApneaNums_ = 0

    with torch.no_grad():
        logger.info("*************** test ***************")
        for X, y in tqdm(test_iter):
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)
            ArousalPred, ApneaPred = net(X)
            ArousalLabel = y[:, 0, :].contiguous()
            ApneaLabel = y[:, 1, :].contiguous()
            ArousalLoss, _ = criterion(ArousalPred, ArousalLabel)
            ApneaLoss, _ = criterion(ApneaPred, ApneaLabel)
            loss = ArousalLoss+ApneaLoss
            total_loss_test += loss.item()

            f1sc, RecallTest, PrecisionTest= Accuracy(ArousalLabel, ArousalPred, "test")
            if RecallTest != -1:
                TestArousalRecord["f1_score"] += f1sc
                TestArousalRecord["recall"] += RecallTest
                TestArousalRecord["precision"] += PrecisionTest
                ArousalNums_ += 1
            TestArousalRecord["loss"] += ArousalLoss.item()

            f1sc, RecallTest, PrecisionTest= Accuracy(ApneaLabel, ApneaPred, "test")
            if RecallTest != -1:
                TestApneaRecord["f1_score"] += f1sc
                TestApneaRecord["recall"] += RecallTest
                TestApneaRecord["precision"] += PrecisionTest
                ApneaNums_ += 1
            TestApneaRecord["loss"] += ApneaLoss.item()
            nums_ += 1
    
    TestArousalRecord["f1_score"] /= ArousalNums_
    TestArousalRecord["recall"] /= ArousalNums_
    TestArousalRecord["precision"] /= ArousalNums_
    TestArousalRecord["loss"] /= nums_

    TestApneaRecord["f1_score"] /= ApneaNums_
    TestApneaRecord["recall"] /= ApneaNums_
    TestApneaRecord["precision"] /= ApneaNums_
    TestApneaRecord["loss"] /= nums_

    total_loss_test /= nums_


    logger.info("\nTest mode: Total loss:{} \nArousal >> F1_score:{}, recall:{}, precision{}, loss:{} \nApnea >> F1_score:{}, recall:{}, precision{}, loss:{}".format(round(total_loss_test, 2),
                    round(TestArousalRecord["f1_score"], 2), round(TestArousalRecord["recall"], 2), round(TestArousalRecord["precision"], 2), round(TestArousalRecord["loss"], 2),
                    round(TestApneaRecord["f1_score"], 2), round(TestApneaRecord["recall"],2), round(TestApneaRecord["precision"], 2), round(TestApneaRecord["loss"], 2)))
    logger.info("************************************\n")

    return TestArousalRecord, TestApneaRecord, total_loss_test, round(TestArousalRecord["f1_score"], 2), round(TestApneaRecord["f1_score"], 2)



BATCH_SIZE = 8
NUM_EPOCHS = 80
NUM_CLASSES = 1
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
# NUM_PRINT = 11000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    WeightDataPath = r"weight\Arousal_Apnea\Train_1206"
    """ Train Dataset and Validation Dataset (104, 105, 107)"""
    # train104_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\104\Apnea\5min5ch0926\Train"
    # label104_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\104\Apnea\5min5ch0926\Label"
    # train105_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\105\Apnea\5min5ch0926\Train"
    # label105_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\105\Apnea\5min5ch0926\Label"
    # train107_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\107\Apnea\5min5ch0926_Clean\Train"
    # label107_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\107\Apnea\5min5ch0926_Clean\Label"

    # Dataset105 = UnetDataset(rootX = train105_datapath, rooty = label105_datapath,
    #                   transform=None)
    # # Dataset107 = UnetDataset(rootX = train107_datapath, rooty = label107_datapath,
    # #                   transform=None)
    # Dataset104 = UnetDataset(rootX = train104_datapath, rooty = label104_datapath,
    #                   transform=None)
    # train_size = int(len(Dataset105)*0.8)
    # test_size = len(Dataset105)-train_size
    # trainset105, valset = torch.utils.data.random_split(Dataset105, [train_size, test_size])
    # trainset = ConcatDataset([Dataset104, Dataset105])
    # trainset = Dataset104
    # valset = Dataset107

    """ Physionet2018 """
    # TrainDatasetPath = r"D:\Joseph_NCHU\Lab\data\Physionet\Arousal\Data1029\Train"
    # LabelDatasetPath = r"D:\Joseph_NCHU\Lab\data\Physionet\Arousal\Data1029\Label"
    # allDataset = UnetDataset(rootX = TrainDatasetPath, rooty = LabelDatasetPath,
    #                   transform=None)
    # train_size = int(len(allDataset)*0.8)
    # test_size = len(allDataset)-train_size
    # trainset, valset = torch.utils.data.random_split(allDataset, [train_size, test_size])
    
    """ Normal, Mild, Moderate, Severe(104, 105) """
    Normal_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Normal"
    Mild_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Mild"
    Moderate_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Moderate"
    Severe_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\ArousalApneaData\Severe"

    Normal_dataset = UnetDataset(rootX = os.path.join(Normal_datapath, "Train"), rooty = os.path.join(Normal_datapath, "Label"), transform=None)
    train_size = int(len(Normal_dataset)*0.8)
    test_size = len(Normal_dataset)-train_size
    Normal_trainset, Normal_valset = torch.utils.data.random_split(Normal_dataset, [train_size, test_size])

    Mild_dataset = UnetDataset(rootX = os.path.join(Mild_datapath, "Train"), rooty = os.path.join(Mild_datapath, "Label"), transform=None)
    train_size = int(len(Mild_dataset)*0.8)
    test_size = len(Mild_dataset)-train_size
    Mild_trainset, Mild_valset = torch.utils.data.random_split(Mild_dataset, [train_size, test_size])

    Moderate_dataset = UnetDataset(rootX = os.path.join(Moderate_datapath, "Train"), rooty = os.path.join(Moderate_datapath, "Label"), transform=None)
    train_size = int(len(Moderate_dataset)*0.8)
    test_size = len(Moderate_dataset)-train_size
    Moderate_trainset, Moderate_valset = torch.utils.data.random_split(Moderate_dataset, [train_size, test_size])

    Severe_dataset = UnetDataset(rootX = os.path.join(Severe_datapath, "Train"), rooty = os.path.join(Severe_datapath, "Label"), transform=None)
    train_size = int(len(Severe_dataset)*0.8)
    test_size = len(Severe_dataset)-train_size
    Severe_trainset, Severe_valset = torch.utils.data.random_split(Severe_dataset, [train_size, test_size])

    trainset = ConcatDataset([Normal_trainset, Mild_trainset, Moderate_trainset, Severe_trainset])
    valset = ConcatDataset([Normal_valset, Mild_valset, Moderate_valset, Severe_valset])




    trainloader = DataLoader(dataset=trainset,
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        drop_last=True)
    
    valloader = DataLoader(dataset=valset,
                        batch_size=BATCH_SIZE, 
                        shuffle=False,
                        drop_last=True)
    
    logger = get_logger(f'{WeightDataPath}\TrainingNote.log')
    model_mod = UnetLSTMModel.ArousalApneaUENNModel(size=5*60*100, num_class=1, n_features=8)
    with open(f'{WeightDataPath}\Model.log', 'w', encoding='utf-8-sig') as f:
        report = summary(model_mod, input_size=(8, 8, 5*60*100), device=DEVICE)
        f.write(str(report))
    logger.info(f"Batch size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}, Loss: CrossEntropy, Optimizer: RMSProp, Device: {DEVICE}")
    model_mod.to(DEVICE)
    if torch.cuda.device_count() > 1:
        model_mod = DataParallel(model_mod)
    train(model_mod, DEVICE, NUM_EPOCHS, LEARNING_RATE, trainloader, valloader, logger, WeightDataPath)


if __name__ == "__main__":
    main()