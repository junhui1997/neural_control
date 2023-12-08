from torch.utils.data import DataLoader
import os
import sys
sys.path.append("..")  # 为了导入上一级目录的 # https://zhuanlan.zhihu.com/p/64893308
from data_provider.data_loader_rbf import Dataset_neural_pd
from test_model.all_func import visual_all, write_info
from test_model.dnn import lstm_p, bp_p
from test_model import iTransformer, FiLM, iTransformer_f
import torch.nn as nn
import numpy as np
import torch

import argparse


# online training
parser = argparse.ArgumentParser(description='neural_pid')
parser.add_argument('--d_model', type=int, default=48, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=3, help='prediction length')
parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')  ###
parser.add_argument('--buffer_size', type=int, default=200, help='batch size')
parser.add_argument('--minimal_size', type=int, default=128, help='should be larger than batch size,and control weather start training')
parser.add_argument('--c_out', type=int, default=4, help='encoder input size')
parser.add_argument('--lr', type=float, default=0.000001, help='learning rate')
parser.add_argument('--dropout', type=int, default=0, help='dropout')
parser.add_argument('--sample_type', type=str, default='log', help='sample type from replay buffer:[linear,log,random,single]')  # single 还没写
parser.add_argument('--input_type', type=str, default='actual', help='ref or actual')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--show_plot', type=int, default=0, help='show plot or not')
parser.add_argument('--model', type=str, default='lstm', help='type of model')

# useless
parser.add_argument('--embed', type=str, default='none', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--d_layers', type=int, default=1, help='useless')
parser.add_argument('--dec_in', type=int, default=7, help='useless')
parser.add_argument('--freq', type=str, default='h', help='useless')
parser.add_argument('--output_attention', action='store_true', help='useless')
parser.add_argument('--distil', action='store_false', help='useless', default=True)
# for attention based model
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
# for timesNet
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
# for serial decompose
parser.add_argument('--moving_avg', type=int, default=5, help='window size of moving average')
args = parser.parse_args()
setting = '{}_dm{}_el{}_sl{}_pl{}_co{}_bs{}_bfs{}_lr{}_st{}'.format(
    args.model,
    args.d_model,
    args.e_layers,
    args.seq_len,
    args.pred_len,
    args.c_out,
    args.batch_size,
    args.buffer_size,
    args.lr,
    args.sample_type,
)

folder_path = 'test_record/rbf/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
#
model_dic = {'lstm': lstm_p,
             'bp': bp_p,
             'it': iTransformer.Model,
             'itf': iTransformer_f.Model,
             'FiLM': FiLM.Model}
# 使用批训练方式
dataset = Dataset_neural_pd(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

net = model_dic[args.model](args).float().cuda()
# 现在是一对一进行的映射

# 定义优化器和损失函数
optim = torch.optim.AdamW(net.parameters(net), lr=args.lr)
Loss = nn.MSELoss()
# Loss = nn.L1Loss()
best_loss = 100
# 下面开始训练：
# 一共训练 1000次
for epoch in range(30):
    train_loss = []
    # print(epoch)
    for batch_x, batch_y in dataloader:
        batch_y = batch_y.cuda().float()
        batch_x = batch_x.cuda().float()
        y_predict = net(batch_x)
        loss = Loss(y_predict, batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
    # 每100次 的时候打印一次日志
    total_loss = np.average(train_loss)
    if (epoch + 1) % 1 == 0:
        print("step: {0} , loss: {1}".format(epoch + 1, total_loss))

    # 这里写的不对吧？？
    if total_loss < best_loss:
        # print('save model')
        best_loss = total_loss
        torch.save(net.state_dict(), folder_path + 'best.pth')

print('best loss is {}'.format(best_loss))
# 使用训练好的模型进行预测
net.load_state_dict(torch.load(folder_path + 'best.pth'))
net.eval()
dataset = Dataset_neural_pd(args)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
all_pred = np.array([[] for i in range(args.c_out)])
all_true = np.array([[] for i in range(args.c_out)])
with torch.no_grad():
    for batch_x, batch_y in dataloader:
        batch_y = batch_y.cuda().float()
        batch_x = batch_x.cuda().float()
        y_predict = net(batch_x)
        # 收集所有的batch_y和y_pred
        y_predict = y_predict.detach().cpu().numpy()[:, -1, :].transpose()  # c_out,batch_size
        batch_y = batch_y.detach().cpu().numpy()[:, -1, :].transpose()

        all_pred = np.concatenate((all_pred, y_predict), axis=1)
        all_true = np.concatenate((all_true, batch_y), axis=1)

all_pred = all_pred.transpose()
all_true = all_true.transpose()

ep_mse = np.mean(np.square(all_pred - all_true))
ep_rms = np.sqrt(np.mean((all_pred - all_true) ** 2))
ep_rms2 = torch.nn.functional.mse_loss(torch.tensor(all_pred), torch.tensor(all_true)) # 理论上和最后一轮的loss一样
print(ep_rms2, ep_mse)
info = 'eq{}_edq{}_ep{}_'.format(0, 0, ep_rms)
write_info("test_record/offline_rbf.txt", setting, info)
print(setting, info)
# 绘图展示预测的和真实数据之间的差异
visual_all(all_true, all_pred, folder_path + 'neural.png', args.c_out, args.show_plot)
