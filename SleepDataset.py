import os
import numpy as np

from torch.utils.data import Dataset, DataLoader

class SleepDataset(Dataset):
    def __init__(self, rootX, rooty, transform):
        ##############################################
        ### Initialize paths, transforms, and so on
        ### data list -> DataFrame ID, Label
        ##############################################
        self.transform = transform
        # load image path and annotations
        self.rootX = rootX
        self.rooty = rooty
        self.imgs = os.listdir(rootX)
        self.lbls = os.listdir(rooty)
        assert len(self.imgs) == len(self.lbls), 'mismatched length!'
        # print('Total data split: {}'.format(len(self.imgs)))
        
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)

        ##############################################
        img = np.load(os.path.join(self.rootX, self.imgs[index]))
        lbl = np.load(os.path.join(self.rooty, self.lbls[index]))
        time = int(str(self.imgs[index]).split("_")[1])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl
        
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.imgs)
    