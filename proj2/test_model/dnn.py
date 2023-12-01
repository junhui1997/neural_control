import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
"""
input:【batch_size,seq_len,dim]
output:[batch_size,seq_len,dim]
对于本例input和output都是d_model
input dim:就是dim
hidden_dim:输出线性层中的尺寸,也是lstm中d_model的尺寸
layer_dim:lstm层数
output_dim:分类的个数
"""

# 一直出现0是网络问题，设置成这样更合理
class lstm_n(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
    # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    def __init__(self, configs=None):
        super().__init__()
        self.input_dim = configs.enc_in  #configs.d_model
        self.hidden_dim = 32
        self.layer_dim = configs.e_layers
        self.out_dim = configs.c_out

        # 注意这里设置了batch_first所以第一个维度是batch，lstm第二个input是输出的维度，第三个是lstm的层数
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(self.hidden_dim * configs.seq_len, self.hidden_dim)
        self.projection2 = nn.Linear(self.hidden_dim , self.out_dim)
        #self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.enc_in)
        init.xavier_uniform_(self.projection.weight)
        init.xavier_uniform_(self.projection2.weight)


    def forward(self, x):
        # init_hidden并不是魔法函数，是每次循环时候手动执行更新的
        # https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        out,  _ = self.lstm(x)
        out = self.act(out)
        out = self.dropout(out)
        out = out.reshape(out.shape[0], -1)
        out = self.projection(out)
        out = self.projection2(out)
        return out

