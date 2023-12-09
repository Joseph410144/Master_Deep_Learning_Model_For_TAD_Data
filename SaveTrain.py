from UnetModel import Unet, USleepMod
from torch import optim
import torch.nn as nn
import torch
from torchvision import transforms
from DatasetUnet import UnetDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torchvision

figIndex = 0
def Accuracy(gt, predict, mode):
    # input data has batch dim
    # label per second
    global figIndex
    acc = 0
    for i in range(len(gt)):
        # label data
        ansdata = []
        LabelData = np.squeeze(gt[i])
        LabelData = LabelData.astype(int)
        Allsec = len(LabelData)//200
        for sec in range(Allsec):
            data = LabelData[sec*200:(sec+1)*200]
            static = np.bincount(data)
            if len(static) == 1:
                ansdata.append(0)
            elif len(static) == 2:
                ansdata.append(np.argmax(static))

        # predict data
        preddata = []
        TestData = np.squeeze(predict[i])
        TestData[TestData >= 0.5] = 1
        TestData[TestData < 0.5] = 0
        TestData = TestData.astype(int)
        for sec in range(Allsec):
            data = TestData[sec*200:(sec+1)*200]
            static = np.bincount(data)
            if len(static) == 1:
                preddata.append(0)
            elif len(static) == 2:
                preddata.append(np.argmax(static))
        correct = sum(np.array(preddata) == np.array(ansdata))
        acc += correct/len(preddata)
        if mode == "test":
            plt.figure()
            plt.subplot(211)
            plt.plot(ansdata)
            plt.subplot(212)
            plt.plot(preddata)
            plt.savefig(rf"D:\Joseph_NCHU\Lab\1DUnet\figure\test0615\{figIndex}_test.jpg")
            figIndex += 1

    return (acc/len(gt))*100

def train(net, device, epochs, lr, train_loader, test_loader):

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCELoss()
    best_loss = float('inf')
    bestTestAcc = 0
    loss_fig = []
    loss_test_fig = []
    acc_fig = []
    acc_test_fig = []
    for epoch in range(epochs):
        accAverage = 0
        lossAverage = 0
        nums = 0
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        print(f"Epoch:{epoch}")
        for image, label in tqdm(train_loader):
            nums+=1
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)       
            # 计算loss
            loss = criterion(pred, label)
            
            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()
            acc_ = Accuracy(label, pred, "train")
            accAverage += acc_
            
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), r'D:\Joseph_NCHU\Lab\1DUnet\weight\Train_0615_Conv\best_model1.pth')
            # 更新参数
            loss.backward()
            # record loss
            lossAverage+=loss.item()
            print(acc_, lossAverage//nums)
            optimizer.step()

        accAverage = accAverage/nums
        lossAverage = lossAverage/nums
        print('\nLoss/train', lossAverage)
        print("Acc:", accAverage)
        acc_fig.append(accAverage)
        loss_fig.append(lossAverage)
        testAcc, testLoss = test(net, test_loader, criterion, device)
        loss_test_fig.append(testLoss)
        acc_test_fig.append(testAcc)
        if testAcc > bestTestAcc:
            torch.save(net.state_dict(), rf'D:\Joseph_NCHU\Lab\1DUnet\weight\Train_0615_Conv\model{epoch}_{testAcc}.pth')

    plt.figure()
    plt.ylim(0, 1)
    plt.title("Train Val accuracy by sec")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    l1 = plt.plot(acc_fig)
    l2 = plt.plot(acc_test_fig)
    plt.legend(handles = [l1, l2], loc='upper right')
    plt.savefig(r"D:\Joseph_NCHU\Lab\1DUnet\weight\Train_0615_Conv\acc_fig.jpg")
    plt.close()


    loss0 = np.array(loss_fig)
    loss1 = np.array(loss_test_fig)
    plt.figure()
    plt.ylim(bottom=0)
    plt.title("Train vs Test loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    l1 = plt.plot(loss0)
    l2 = plt.plot(loss1)
    plt.legend(handles = [l1, l2], loc='upper right')
    plt.savefig(r"D:\Joseph_NCHU\Lab\1DUnet\weight\Train_0615_Conv\loss_fig.jpg")
    plt.close()

def test(net, test_iter, criterion, device):
    total = 0
    test_acc = 0
    total_loss = 0
    net.eval()

    with torch.no_grad():
        print("*************** test ***************")
        for X, y in tqdm(test_iter):
            total += 1
            X, y = X.to(device=device, dtype=torch.float32), y.to(device=device, dtype=torch.float32)

            output = net(X)
            loss = criterion(output, y)
            total_loss += loss.item()
            pred = output.data.cpu().numpy()
            label = y.data.cpu().numpy()
            accT = Accuracy(label, pred, "test")
            test_acc += accT
            
    test_acc = test_acc / total
    total_loss = total_loss/total

    print("test_loss: {:.3f} | test_acc: {:6.3f}%"
          .format(total_loss, test_acc))
    print("************************************\n")
    net.train()

    return test_acc, total_loss



BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_CLASSES = 5
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
# NUM_PRINT = 11000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\104\6chTest1min\train"
    label_datapath = r"D:\Joseph_NCHU\Lab\data\北醫UsleepData\104\6chTest1min\label"

    model_mod = USleepMod.USleep_1min(size=60*200, channels=6, num_class=1)
    allDataset = UnetDataset(rootX = train_datapath, rooty = label_datapath,
                      transform=None)
    train_size = int(len(allDataset)*0.7)
    test_size = len(allDataset) - train_size

    trainset, valset = torch.utils.data.random_split(allDataset, [train_size, test_size])
    trainloader = DataLoader(dataset=trainset,
                        batch_size=8, 
                        shuffle=False,
                        drop_last=True)
    valloader = DataLoader(dataset=valset,
                        batch_size=8, 
                        shuffle=False)
    images, labels = next(iter(trainloader))
    print(images.shape)
    print(images.shape[0])
    print(labels.shape)
    print(DEVICE)
    model_mod.to(DEVICE)
    train(model_mod, DEVICE, NUM_EPOCHS, LEARNING_RATE, trainloader, valloader)


if __name__ == "__main__":
    main()