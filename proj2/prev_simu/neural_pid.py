import matlab.engine
import numpy
import numpy as np
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from test_model.dnn import lstm_n
from test_model.block import ReplayBuffer
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.io
import os
parser = argparse.ArgumentParser(description='neural_pid')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

args = parser.parse_args()
setting = 'dm{}_el{}_sl{}'.format(
    args.d_model,
    args.e_layers,
    args.seq_len
)
folder_path = './results/' + setting + '/'



# initialize model
model = lstm_n(args).float()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.train()

# for online learning
buffer_size = 1000
minimal_size = 64
batch_size = 16
replay_buffer = ReplayBuffer(buffer_size)

# for online training
# while True:
#     sys_in = np.random.rand(args.seq_len, args.enc_in)
#     current_label = np.random.rand(1, args.enc_in)  #现在是输出下一个step，直觉上来说预测下一到两个会更好一点
#     replay_buffer.add(sys_in, current_label)
#     if replay_buffer.size() > minimal_size:
#         inputs, labels = replay_buffer.sample(batch_size)
#         inputs = torch.from_numpy(inputs).to(torch.float32)
#         # optimizer.zero_grad()
#         # outputs = model(sys_in)
#         # outputs = outputs.squeeze(0)
#         # o_py = outputs.tolist()
#         # print('training', outputs)
#         # loss = F.mse_loss(outputs, torch.zeros(1, args.seq_len, args.enc_in))
#         # loss.backward()
#         # optimizer.step()

while True:
    sys_in = torch.from_numpy(np.random.rand(args.seq_len, 2)).unsqueeze(0).to(torch.float32)
    sys_in.requires_grad = True
    optimizer.zero_grad()
    outputs = model(sys_in)
    outputs = outputs.squeeze(0)
    o_py = outputs.tolist()
    loss = F.mse_loss(sys_in, torch.zeros(1, args.seq_len, args.enc_in))
    print('training', outputs, loss)
    loss.backward()
    optimizer.step()

# while True:
#     model.train()
#     sys_in = torch.from_numpy(np.random.randn(args.seq_len, 2)).unsqueeze(0).to(torch.float32)
#     #sys_in.requires_grad = True
#     optimizer.zero_grad()
#     outputs = model(sys_in)
#     outputs = outputs.squeeze(0)
#     o_py = outputs.tolist()
#     print('training', sys_in[0, 0, :], outputs)
#     loss = F.mse_loss(outputs, torch.zeros(1, args.seq_len, args.enc_in))
#     loss.backward()
#     optimizer.step()
#     model.eval()

# load baseline data
mat_g = scipy.io.loadmat('gravity.mat')
mat_og = scipy.io.loadmat('gravity_offset.mat')
q_g = mat_g['q']
dq_g = mat_g['dq']
e1_g = mat_g['e1']
sTau_g = mat_g['sTau']
q_og = mat_og['q']
dq_og = mat_og['dq']
e1_og = mat_og['e1']
sTau_og = mat_og['sTau']

# sys config
counter = 1
sample_rate = 0.0025
total_t = 32
pi = numpy.pi
# 一定要注意data type问题
o_py = [0, 0]
o_pys = np.array([[0, 0]])
engine = matlab.engine.start_matlab()
print('sys start')
engine.neural_pid(nargout=0)




torch.cuda.empty_cache()

while True:
    print('simulation counter:', counter)
    engine.set_param('Code_2DoF_Simulation', 'SimulationCommand', 'pause', nargout=0)
    sys_in = engine.eval('e1')
    #print(sys_in)
    if len(sys_in) < 20:
        pass
    else:
        sys_in = torch.from_numpy(np.array(sys_in)[-20:, :]).unsqueeze(0).to(torch.float32)
        sys_in.requires_grad = True
        model.train()
        optimizer.zero_grad()
        outputs = model(sys_in)
        loss = F.mse_loss(sys_in, torch.zeros(1, args.seq_len, args.enc_in))
        loss.backward()
        optimizer.step()
        outputs = outputs.squeeze(0)
        o_py = outputs.tolist()
        print(o_py)
        o_pys = np.concatenate((o_pys, np.array(o_py).reshape(1, 2)), axis=0)
        engine.workspace['o_py'] = matlab.double(o_py)

    if counter % 100 == 0:
        # plot every x iternation
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # extract sys data
        zeros = np.zeros((int(total_t/sample_rate-counter+1), args.enc_in))
        e1 = np.concatenate((np.array(engine.eval('e1')), zeros), axis=0)
        q = np.concatenate((np.array(engine.eval('q')), zeros), axis=0)
        sTau = np.concatenate((np.array(engine.eval('sTau')), zeros), axis=0)
        dq = np.concatenate((np.array(engine.eval('dq')), zeros), axis=0)
        t1 = np.arange(0, 32.0025, 0.0025)
        t1 = t1.reshape(-1, 1)
        tout = np.linspace(0, 32, num=len(t1))
        # save all data every 200 iternation
        if counter % 200 == 0:
            savemat(folder_path+'output_{}.mat'.format(counter), {'e1': e1,
                                                  'q': q,
                                                  'sTau': sTau,
                                                  'dq': dq,
                                                  'o_pys': o_pys})
        # plot the first graph
        plt.figure(1)
        plt.subplot(321)
        plt.plot(tout, np.sin(np.pi / 4 * tout) - np.pi / 2, 'r', label='gt')
        plt.plot(tout, q[:, 0],  linewidth=1.0, label='ours')
        plt.plot(tout, q_g[:, 0],  linewidth=1.0, label='g')
        plt.plot(tout, q_og[:, 0],  linewidth=1.0, label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$q_1$ (rad)')

        plt.subplot(323)
        plt.plot(t1, e1[:, 0],  linewidth=1.0, label='our')
        plt.plot(t1, e1_g[:, 0], linewidth=1.0, label='g')
        plt.plot(t1, e1_og[:, 0], linewidth=1.0, label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$e_{q_1}$ (deg)')

        plt.subplot(322)
        plt.plot(tout, np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)), 'r')
        plt.plot(tout, q[:, 1], linewidth=1.0, label='our')
        plt.plot(tout, q_g[:, 1], linewidth=1.0, label='g')
        plt.plot(tout, q_og[:, 1], linewidth=1.0, label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$q_2$ (rad)')

        plt.subplot(324)
        plt.plot(tout, np.abs(np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)) - q[:, 1]) * 180/np.pi,  linewidth=1.0, label='our')
        plt.plot(tout, np.abs(np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)) - q_g[:, 1]) * 180 / np.pi, linewidth=1.0, label='g')
        plt.plot(tout, np.abs(np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2)) - q_og[:, 1]) * 180 / np.pi, linewidth=1.0, label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$e_{q_2}$ (deg)')

        plt.subplot(325)
        plt.plot(t1, sTau[:, 0],  linewidth=1.0, label='our')
        plt.plot(t1, sTau_g[:, 0], linewidth=1.0, label='g')
        plt.plot(t1, sTau_og[:, 0],  linewidth=1.0, label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$\tau_1$ [Nm]')

        plt.subplot(326)
        plt.plot(t1, sTau[:, 1], linewidth=1.0, label='our')
        plt.plot(t1, sTau_g[:, 1],  linewidth=1.0, label='g')
        plt.plot(t1, sTau_og[:, 1],  linewidth=1.0, label='og')
        plt.ylabel(r'$\tau_2$ [Nm]')
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig(folder_path + 'pos_{}.svg'.format(counter), format='svg')
        plt.clf()

        # plot the second graph
        plt.figure(2)
        plt.subplot(221)
        t = tout
        plt.plot(tout, pi / 4 * np.cos(pi / 4 * tout), 'r')
        plt.plot(tout, dq[:, 0],  label='our')
        plt.plot(tout, dq_g[:, 0],  label='g')
        plt.plot(tout, dq_og[:, 0],  label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$\dot{q}_1$ (rad/s)')

        plt.subplot(223)
        plt.plot(tout, np.abs(pi / 4 * np.cos(pi / 4 * tout) - dq[:, 0]), 'k', linewidth=1.0, label='our')
        plt.plot(tout, np.abs(pi / 4 * np.cos(pi / 4 * tout) - dq_g[:, 0]), 'k', linewidth=1.0, label='g')
        plt.plot(tout, np.abs(pi / 4 * np.cos(pi / 4 * tout) - dq_og[:, 0]), 'k', linewidth=1.0, label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$e\dot{q}_1$ (rad/s)')

        plt.subplot(222)
        plt.plot(tout, pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
            pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
            pi / 4 * (t + np.pi / 2)), 'r')
        plt.plot(tout, dq[:, 1], label='our')
        plt.plot(tout, dq_g[:, 1], label='g')
        plt.plot(tout, dq_og[:, 1], label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$\dot{q}_2$ (rad/s)')

        plt.subplot(224)
        plt.plot(tout, np.abs(pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
            pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
            pi / 4 * (t + np.pi / 2)) - dq[:, 1]),  linewidth=1.0, label='our')
        plt.plot(tout, np.abs(pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
            pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
            pi / 4 * (t + np.pi / 2)) - dq_g[:, 1]),  linewidth=1.0, label='g')
        plt.plot(tout, np.abs(pi / 2 * -pi / 8 * np.sin(pi / 4 * (t + np.pi / 2)) * np.sin(
            pi / 8 * (t + np.pi / 2)) + pi / 2 * pi / 4 * np.cos(pi / 8 * (t + np.pi / 2)) * np.cos(
            pi / 4 * (t + np.pi / 2)) - dq_og[:, 1]),  linewidth=1.0, label='og')
        plt.legend(loc='upper right', fontsize='small')
        plt.ylabel(r'$e\dot{q}_2$ (rad/s)')
        plt.savefig(folder_path + 'vel_{}.svg'.format(counter), format='svg')
        plt.clf()

        plt.figure(3)
        plt.plot(o_pys[:, 0], label='o1')
        plt.plot(o_pys[:, 1], label='o2')
        plt.legend(loc='upper right', fontsize='small')
        plt.savefig(folder_path + 'o_{}.svg'.format(counter), format='svg')
        plt.clf()
        plt.close()

    engine.set_param('Code_2DoF_Simulation', 'SimulationCommand', 'update', nargout=0)
    engine.set_param('Code_2DoF_Simulation', 'SimulationCommand', 'step', nargout=0)
    if counter >= total_t / sample_rate:
        engine.set_param('Code_2DoF_Simulation', 'SimulationCommand', 'stop', nargout=0)
        break
    counter = counter + 1
engine.quit() # quit Matlab engine