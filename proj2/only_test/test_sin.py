from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch
import torch.nn.init as init
import torch.nn.functional as F

# 准备数据
# x.shape(400) y.shape(400)
x = np.linspace(-2 * np.pi, 2 * np.pi, 400)
y = np.sin(x)
# 将数据做成数据集的模样
# X[400,1] Y[400,1]
X = np.expand_dims(x, axis=1)
Y = y.reshape(400, -1)
# 使用批训练方式
dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)


# 神经网络主要结构，这里就是一个简单的线性结构


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=10), nn.ReLU(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, input: torch.FloatTensor):
        return self.net(input)


class Net2(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
    # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    def __init__(self):
        super().__init__()
        self.input_dim = 1 #configs.enc_in  #configs.d_model
        self.hidden_dim = 64
        self.layer_dim = 3
        self.out_dim = 2

        # 注意这里设置了batch_first所以第一个维度是batch，lstm第二个input是输出的维度，第三个是lstm的层数
        self.lstm = nn.GRU(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True) #LSTM
        # self.act = F.gelu #F.gelu
        # self.dropout = nn.Dropout(0.0001)
        # self.linaer = nn.Linear(self.hidden_dim, 1)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=10), nn.ReLU(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )


    def forward(self, x):
        # init_hidden并不是魔法函数，是每次循环时候手动执行更新的
        # https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        out,  _ = self.lstm(x)
        out = self.linear(out)
        return out
net = Net2().cuda()
# 现在是一对一进行的映射

# 定义优化器和损失函数
optim = torch.optim.Adam(Net.parameters(net), lr=0.001)
Loss = nn.MSELoss()

# 下面开始训练：
# 一共训练 1000次
for epoch in range(1000):
    loss = None
    for batch_x, batch_y in dataloader:
        batch_y = batch_y.cuda()
        batch_x = batch_x.cuda()
        y_predict = net(batch_x)
        loss = Loss(y_predict, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # 每100次 的时候打印一次日志
    if (epoch + 1) % 100 == 0:
        print("step: {0} , loss: {1}".format(epoch + 1, loss.item()))

# 使用训练好的模型进行预测
predict = net(torch.tensor(X, dtype=torch.float).cuda())

# 绘图展示预测的和真实数据之间的差异
import matplotlib.pyplot as plt

plt.plot(x, y, label="fact")
plt.plot(x, predict.cpu().detach().numpy(), label="predict")
plt.title("sin function")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend()
plt.savefig(fname="result.png")
plt.show()

# linear
# step: 100 , loss: 0.06755948066711426
# step: 200 , loss: 0.003788222325965762
# step: 300 , loss: 0.0004728269996121526
# step: 400 , loss: 0.0001810075482353568
# step: 500 , loss: 0.0001108720971387811
# step: 600 , loss: 6.29749265499413e-05
# step: 700 , loss: 3.707894938997924e-05
# step: 800 , loss: 0.0001250380591955036
# step: 900 , loss: 3.0654005968244746e-05
# step: 1000 , loss: 4.349641676526517e-05

# lstm layer 1, relu和gelu区别不大
# step: 100 , loss: 0.22633373737335205
# step: 200 , loss: 0.04226946830749512
# step: 300 , loss: 0.03057619370520115
# step: 400 , loss: 0.022798582911491394
# step: 500 , loss: 0.01327531784772873
# step: 600 , loss: 0.007136711850762367
# step: 700 , loss: 0.0030499889981001616
# step: 800 , loss: 0.004619299899786711
# step: 900 , loss: 0.002987706335261464
# step: 1000 , loss: 0.0013432997511699796


# lstm layer 2, 双层lstm提升巨大
# dropout 0.1 时候模型抖动很大, 需要一直缩减到0.0001
# bp可以抑制开头时候的抖动，lstm模型在开头的时候有个很大的抖动不知道为啥
# step: 100 , loss: 0.040910620242357254
# step: 200 , loss: 0.003032656852155924
# step: 300 , loss: 0.0023330519907176495
# step: 400 , loss: 0.00028826436027884483
# step: 500 , loss: 0.0013971570879220963
# step: 600 , loss: 0.0008668128866702318
# step: 700 , loss: 0.0008832477033138275
# step: 800 , loss: 0.0001414590806234628
# step: 900 , loss: 0.00039753204328007996
# step: 1000 , loss: 0.0006103190826252103

# gru有时候会出现奇怪的结果