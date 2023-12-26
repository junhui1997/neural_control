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

class lstm_p(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    # num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs
    # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
    def __init__(self, configs=None):
        super().__init__()
        self.input_dim = configs.enc_in  #configs.d_model
        self.hidden_dim = configs.d_model
        self.layer_dim = configs.e_layers
        self.out_dim = configs.c_out

        # 注意这里设置了batch_first所以第一个维度是batch，lstm第二个input是输出的维度，第三个是lstm的层数
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection_l = nn.Linear(configs.seq_len, configs.pred_len)
        self.projection_dim = nn.Linear(self.hidden_dim , self.out_dim)
        #self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.enc_in)
        init.xavier_uniform_(self.projection_l.weight)
        init.xavier_uniform_(self.projection_dim.weight)

    # 标准输出 [batch_size, pred_len, c_out]
    # 输入[batch_size,seq_len,d_model]
    def forward(self, x):
        # init_hidden并不是魔法函数，是每次循环时候手动执行更新的
        # https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
        out,  _ = self.lstm(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.projection_l(out.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.projection_dim(out)
        return out


class bp_p(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, configs=None):
        super().__init__()
        self.layer = configs.e_layers
        self.input_l = nn.Linear(configs.seq_len*configs.enc_in, configs.d_model)
        hidden_l = []
        for i in range(self.layer):
            hidden_l.append(nn.Linear(configs.d_model, configs.d_model))
            hidden_l.append(nn.Sigmoid()) #nn.ReLU() # nn.ELU() # nn.Tanh()
        self.bp = nn.ModuleList(hidden_l)
        self.output_l = nn.Linear(configs.d_model, configs.pred_len*configs.c_out)
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
    def forward(self, x):
        # [batch_size,seq_len,enc_in]->[batch_size,seq_len*enc_in]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.input_l(x)
        for i in range(self.layer):
            x = self.bp[i](x)
        x = self.output_l(x)
        x = x.reshape(batch_size, self.pred_len, self.c_out)
        return x