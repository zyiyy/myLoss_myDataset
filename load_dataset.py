# coding=utf-8


import pandas as pd
import torch
from torch.utils.data import DataLoader

# x_train = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168,
#                     9.779, 6.182, 7.59, 2.167, 7.042,
#                     10.791, 5.313, 7.997, 3.1], dtype=np.float32)
# y_train = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573,
#                     3.366, 2.596, 2.53, 1.221, 2.827,
#                     3.465, 1.65, 2.904, 1.3], dtype=np.float32)
#
# data = pd.DataFrame()
# data['x'] = torch.from_numpy(x_train)
# data['y'] = torch.from_numpy(y_train)
#
# data.to_csv("data/data.csv", sep=",", header=True, index=False)


# 定义自己的DataSet, 需要实现__init__(), __getitem__(), __len__()三个方法
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y

    def __len__(self):
        return len(self.X)

batch_size = 15
data = pd.read_csv('./data/data.csv')
X , Y = data['x'], data['y']
train_dataset = MyDataSet(X, Y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
