import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CW2_dataset(Dataset):
    def __init__(self, train=True, fold=None):
        super(CW2_dataset, self).__init__()
        self.x, self.y = getdata(train,fold)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

def shuffle_data(x,y):
    index = torch.randperm(x.size(0))
    x = x[index]
    y = y[index]
    return x,y

def getdata(train,fold):
    df = pd.read_csv('./CW_Data.csv', sep=',', header=0)
    data = df.values
    Class_0 = []
    Class_1 = []
    Class_2 = []
    Class_3 = []
    Class_4 = []
    for i in range(len(data)):
        if data[i, -1] == 0:
            Class_0.append(data[i, 1:-1])
        if data[i, -1] == 1:
            Class_1.append(data[i, 1:-1])
        if data[i, -1] == 2:
            Class_2.append(data[i, 1:-1])
        if data[i, -1] == 3:
            Class_3.append(data[i, 1:-1])
        if data[i, -1] == 4:
            Class_4.append(data[i, 1:-1])