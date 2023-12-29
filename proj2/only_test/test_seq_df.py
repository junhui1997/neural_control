from torch.utils.data import DataLoader
from proj2.data_provider.data_loader_pd_seq import Dataset_neural_pd
from proj2.test_model.all_func import visual_all
import torch.nn as nn
import numpy as np
import torch
import os
import torch.nn.init as init
import torch.nn.functional as F

# 准备数据
enc_in = 13 #12  without additional data
c_out = 4
folder_path = 'test_record/seq/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
# 使用批训练方式
dataset = Dataset_neural_pd()
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)


# 神经网络主要结构，这里就是一个简单的线性结构


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=enc_in, out_features=10), nn.ReLU(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, c_out)
        )

    def forward(self, input: torch.FloatTensor):
        return self.net(input)


class Net2(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
    # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    def __init__(self):
        super().__init__()
        self.input_dim = enc_in # configs.enc_in  #configs.d_model
        self.hidden_dim = 64
        self.layer_dim = 2
        self.out_dim = c_out

        # 注意这里设置了batch_first所以第一个维度是batch，lstm第二个input是输出的维度，第三个是lstm的层数
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)  # LSTM
        # self.act = F.gelu #F.gelu
        # self.dropout = nn.Dropout(0.0001)
        # self.linaer = nn.Linear(self.hidden_dim, 1)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=10), nn.ReLU(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, c_out)
        )

    def forward(self, x):
        # init_hidden并不是魔法函数，是每次循环时候手动执行更新的
        # https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out


net = Net2().cuda()
# 现在是一对一进行的映射

# 定义优化器和损失函数
optim = torch.optim.AdamW(Net.parameters(net), lr=0.001)
Loss = nn.MSELoss()
best_loss = 100
# 下面开始训练：
# 一共训练 1000次
for epoch in range(20):
    loss = None
    #print(epoch)
    for batch_x, batch_y in dataloader:
        batch_y = batch_y.cuda().float()
        batch_x = batch_x.cuda().float()
        y_predict = net(batch_x)
        loss = Loss(y_predict, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    # 每100次 的时候打印一次日志
    if (epoch + 1) % 1 == 0:
        print("step: {0} , loss: {1}".format(epoch + 1, loss.item()))

    if loss.item() < best_loss:
        #print('save model')
        best_loss = loss.item()
        torch.save(net.state_dict(), folder_path+'best.pth')

print('best loss is {}'.format(best_loss))
# 使用训练好的模型进行预测
net.load_state_dict(torch.load(folder_path+'best.pth'))
net.eval()
dataset = Dataset_neural_pd()
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
all_pred = np.array([[] for i in range(c_out)])
all_true = np.array([[] for i in range(c_out)])
for batch_x, batch_y in dataloader:
    batch_y = batch_y.cuda().float()
    batch_x = batch_x.cuda().float()
    y_predict = net(batch_x)
    # 收集所有的batch_y和y_pred
    y_predict = y_predict.detach().cpu().numpy().transpose()  # c_out,batch_size
    batch_y = batch_y.detach().cpu().numpy().transpose()
    all_pred = np.concatenate((all_pred, y_predict), axis=1)
    all_true = np.concatenate((all_true, batch_y), axis=1)

all_pred = all_pred.transpose()
all_true = all_true.transpose()

# 绘图展示预测的和真实数据之间的差异
visual_all(all_true, all_pred, folder_path+'neural_pd_lstm.png', c_out)