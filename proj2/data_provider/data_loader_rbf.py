import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from test_model.all_func import apply_norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Dataset_neural_pd(Dataset):
    def __init__(self, args):
        folder = '../data/'
        # self.df_raw = pd.read_pickle(folder + 'pd_ref_train.pkl')
        self.df_raw = pd.read_pickle(folder + 'pd_rbf.pkl')
        self.df_raw['t'] = [1/12000*i for i in range(len(self.df_raw))]
        self.df_raw = self.df_raw[600:]
        self.len_df = len(self.df_raw)
        self.args = args
        a = 1

    def __getitem__(self, index):
        # # sssssss
        # #     lllpppp
        # # seq_x是输入进encoder的值，seq_y是输入进decoder的值
        # seq_x = self.data_x['value list'].iloc[index].astype('float64')
        # seq_y = self.data_x['label'].iloc[index].astype('float64')
        # seq_x = torch.from_numpy(seq_x)
        # seq_y = torch.from_numpy(seq_y)
        # seq_x_mark = self.data_stamp_x
        # seq_y_mark = self.data_stamp_y

        # data [4,] label [2,]
        data = self.df_raw[['q1', 'q2', 'dq1', 'dq2', 't']].iloc[index:index + self.args.seq_len].to_numpy().astype('float64')   # 'q1', 'q2', 'dq1', 'dq2', 't'
        label = self.df_raw[['eq1', 'eq2', 'edq1', 'edq2']].iloc[index + self.args.seq_len:index +self.args.seq_len+self.args.pred_len].to_numpy().astype('float64')
        label = label * 100
        data = apply_norm(data)
        data = torch.from_numpy(data)  # 展平时候是沿着行的，为了实现p_1,p_2,p_3,v_1,v_2,v_3
        label = torch.from_numpy(label)
        # if index < self.prev_len:
        #     addition_data = torch.tensor([0, 0, 0, 0]).reshape(-1)
        # else:
        #     other_info = self.df_raw[['eq1', 'eq2']].iloc[index - self.prev_len:index].to_numpy().astype('float64')
        #     other_info = np.array([np.mean(other_info, axis=0), np.std(other_info, axis=0)])
        #     addition_data = torch.from_numpy(other_info).reshape(-1)
        # data = np.concatenate((data, addition_data), axis=0)
        # data = np.concatenate((data, np.array([index/self.len_df])), axis=0)
        return data, label

    def __len__(self):
        return len(self.df_raw) - self.args.seq_len - self.args.pred_len
