import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
class Dataset_neural_pd(Dataset):
    def __init__(self, c_out, enc_in):
        folder = '../data/'
        #self.df_raw = pd.read_pickle(folder + 'pd_ref_train.pkl')
        self.df_raw = pd.read_pickle(folder + 'pd_ref_train_4x.pkl')
        self.label_list = ['eq1', 'eq2'][:c_out]
        if enc_in == 2:
            self.data_list = ['q1', 'dq1']
        elif enc_in == 4:
            self.data_list = ['q1', 'q2', 'dq1', 'dq2']
        self.df_raw = self.df_raw #[5000:]
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
        data = self.df_raw[self.data_list].iloc[index].to_numpy().astype('float64')
        label = self.df_raw[self.label_list].iloc[index].to_numpy().astype('float64')
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        return data, label

    def __len__(self):
        return len(self.df_raw)