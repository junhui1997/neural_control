import numpy as np
from test_model.all_func import stiffness, coriolis, gravity, dynamics, disturbance, plot_all, apply_norm, dynamic_adjust, generate_ref,visual_all
from exp.exp_basic import exp_model
import os
import argparse
from test_model.dnn import lstm_n



# system parse
parser = argparse.ArgumentParser(description='neural_pid')
parser.add_argument('--d_model', type=int, default=48, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=4, help='encoder input size')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')   ###
parser.add_argument('--buffer_size', type=int, default=2000, help='batch size')
parser.add_argument('--minimal_size', type=int, default=128, help='should be larger than batch size,and control weather start training')
parser.add_argument('--c_out', type=int, default=4, help='encoder input size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--dropout', type=int, default=0, help='dropout')
parser.add_argument('--sample_type', type=str, default='log', help='sample type from replay buffer:[linear,log,random,single]')  # single 还没写
parser.add_argument('--input_type', type=str, default='ref', help='ref or actual')
args = parser.parse_args()
setting = 'dm{}_el{}_sl{}_co{}'.format(
    args.d_model,
    args.e_layers,
    args.seq_len,
    args.c_out
)
folder_path = './results/online_training/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


exps = exp_model(args,lstm_n)


# sys config
task_name = 'ss'
counter = 0
sample_rate = 0.0025
total_t = 32
total_epoch = int(total_t/sample_rate)
dim =2
torque_c = 300  ##torque limit 130
q_init = np.array([[-1], [0]])
dq_init = np.array([[0], [0]])
t1 = np.arange(0, total_t+sample_rate, sample_rate)
t1 = t1.reshape(-1, 1)
tout = t1
q_ref, dq_ref = generate_ref(task_name, tout)
total_eq = np.array([[], []])
total_edq = np.array([[], []])
total_q = q_init
total_dq = dq_init
total_u = np.array([[0], [0]])
total_pred = np.array([[] for i in range(args.c_out)])
e_predv = np.array([[0], [0]])

#pd controller
kp1 = 1000
kp2 = 800
kd1 = 500
kd2 = 80
kp = np.diag([kp1, kp2])
kd = np.diag([kd1, kd2])
while counter < total_epoch:
    if counter > args.seq_len+1:
        if args.input_type =='actual':
            total_info = np.concatenate((total_q.transpose()[-args.seq_len-1:-1, :], total_dq.transpose()[-args.seq_len-1:-1, :], total_eq.transpose()[-args.seq_len-1:-1, :]), axis=1)[:, :args.enc_in]
        else:
            other_info = np.array([(counter-1)/total_epoch])
            other_info_r = np.tile(other_info, (args.seq_len, 1))
            total_info = np.concatenate((q_ref[counter-args.seq_len:counter, :], dq_ref[counter-args.seq_len:counter, :], other_info_r), axis=1)[:,:args.enc_in]
        #total_info = apply_norm(total_info)
        total_label = np.concatenate((total_eq.transpose()[-1, :], total_edq.transpose()[-1, :]))
        exps.update_buffer(total_info, total_label)
    if exps.get_buffer_size() >= args.minimal_size:
        exps.train_one_epoch()
        if args.input_type == 'actual':
            current_inputs = np.concatenate((total_q.transpose()[-args.seq_len:, :], total_dq.transpose()[-args.seq_len:, :], total_eq.transpose()[-args.seq_len:, :]), axis=1)[:, :args.enc_in]
        else:
            other_info = np.array([(counter)/total_epoch])
            other_info_r = np.tile(other_info, (args.seq_len, 1))
            current_inputs = np.concatenate((q_ref[counter - args.seq_len+1:counter+1, :], dq_ref[counter - args.seq_len+1:counter+1, :], other_info_r), axis=1)[:, :args.enc_in]
        #current_inputs = apply_norm(current_inputs)
        pred = exps.pred(current_inputs)
        pred = dynamic_adjust(pred, counter, 500)  # 大学习率和最开始输出控制
        e_pred = pred[:2, :]
        de_pred = pred[2:, :]
        # de_pred = dynamic_adjust(de_pred, counter, 5000)
        print(e_pred)
        # de_pred = np.array([[0], [0]])
    else:
        e_pred = np.array([[0], [0]])
        de_pred = np.array([[0], [0]])
        pred = np.array([[0] for i in range(args.c_out)])
    e_q_t = q_init - q_ref[counter].reshape(dim, 1)
    e_dq_t = dq_init - dq_ref[counter].reshape(dim, 1)
    e_q = e_q_t + e_pred
    e_dq = e_dq_t + de_pred
    u_pd = -kp@e_q - kd@e_dq
    u_pd = np.clip(u_pd, -torque_c, torque_c)
    M = stiffness(q_init)
    C = coriolis(q_init, dq_init)
    G = gravity(q_init)
    disturb = disturbance(counter * sample_rate)
    TL = u_pd - C - G + disturb
    ddq = dynamics(TL, M)
    dq = dq_init + ddq*sample_rate
    q = q_init + dq*sample_rate
    dq_init = dq
    q_init = q
    counter += 1
    total_eq = np.concatenate((total_eq, e_q_t), axis=1)
    total_edq = np.concatenate((total_edq, e_dq_t), axis=1)
    total_dq = np.concatenate((total_dq, dq), axis=1)
    total_q = np.concatenate((total_q, q), axis=1) # (c_out, total_epoch)
    total_u = np.concatenate((total_u, u_pd), axis=1)
    total_pred = np.concatenate((total_pred, pred), axis=1)
    print(counter)
    # 这样相当于是有重力但是没有disturbance

q = total_q.transpose()
dq = total_dq.transpose()
sTau = total_u.transpose()
all_pred = total_pred.transpose()[args.seq_len:, :]  # 去除掉最开始没有介入网络的部分
all_true = np.concatenate((total_eq, total_edq), axis=0).transpose()[args.seq_len:, :]
visual_all(all_true, all_pred, folder_path+'loss_.png', args.c_out)
tout = np.linspace(0, 32, num=len(t1)) #不使用np速度回很慢
plot_all(q, dq, sTau, tout, q_ref, dq_ref, folder_path, flag=task_name)
