import socket
from struct import pack, unpack
import threading
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from test_model.dnn import lstm_n
import matplotlib.pyplot as plt
from scipy.io import savemat
import os
_send_lock = threading.Lock()

# 创建客户端socket
hostport = ('127.0.0.1', 30000)
cmd_socket = socket.create_connection(hostport, timeout=15)
cmd_socket.settimeout(None)

# 准备要发送的数组
o_py = [0.0, 0.0]
e1_memo = np.array([[]])
q_memo = np.array([[]])
sTau_memo = np.array([[]])
dq_memo = np.array([[]])

# 将数组转换为字节流
x = 1
while True:
    x+=1
    o_py[0] = o_py[0]+x
    data = pack('2d', *o_py)
    with _send_lock:
        # 发送数据
        cmd_socket.send(data)  # 发送数据到套接字
        server_reply = cmd_socket.recv(64)  # 接收套接字数据
    temp = list(unpack("8d", server_reply))
    e1_memo = np.concatenate((e1_memo, np.array([temp[0], temp[1]]).reshape(1, 2)), axis=1) #这个地方写错了，应该是axis = 0但是=0的话，没法concatenate，需要先赋予一个初值
    q_memo = np.concatenate((q_memo, np.array([temp[2], temp[3]]).reshape(1, 2)), axis=1)
    sTau_memo = np.concatenate((sTau_memo, np.array([temp[4], temp[5]]).reshape(1, 2)), axis=1)
    dq_memo = np.concatenate((dq_memo, np.array([temp[6], temp[7]]).reshape(1, 2)), axis=1)
# 关闭连接
cmd_socket.close()
