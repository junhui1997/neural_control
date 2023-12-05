import numpy as np
from test_model.all_func_RBF import Encoder, Controller, Dynamics_W, Dynamics_q, Dynamics_dq, Dynamics_ddq, Dynamics_z, generate_ref, plot_data, generate_info
from test_model.all_func import generate_mat, dynamic_adjust, visual_all, write_info
import os
import argparse
from exp.exp_basic import exp_model
from test_model.dnn import lstm_p
from test_model import iTransformer
import warnings
warnings.filterwarnings("ignore")


# online training
parser = argparse.ArgumentParser(description='neural_pid')
parser.add_argument('--d_model', type=int, default=48, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=3, help='prediction length')
parser.add_argument('--enc_in', type=int, default=4, help='encoder input size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')  ###
parser.add_argument('--buffer_size', type=int, default=500, help='batch size')
parser.add_argument('--minimal_size', type=int, default=128, help='should be larger than batch size,and control weather start training')
parser.add_argument('--c_out', type=int, default=4, help='encoder input size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
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
parser.add_argument('--distil', action='store_false', help='useless',default=True)
# for attention based model
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
# for timesNet
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
# for serial decompose
parser.add_argument('--moving_avg', type=int, default=50, help='window size of moving average')


args = parser.parse_args()
if args.minimal_size > args.buffer_size:
    args.minimal_size = args.buffer_size
if args.buffer_size < args.batch_size:
    args.buffer_size = args.batch_size
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
# ./代表的是当前路径下
folder_path = './results/online_rbf/' + setting + '/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

model_dic = {'lstm': lstm_p,
             'it': iTransformer.Model}

exps = exp_model(args, model_dic[args.model])

# controller
Number_All = 60001
Number_Major = 12001
T_final = 30
count = 1
i = 0  # python的i取值要注意 change
T = 0
dt = 0.0005
nodes = 13
q = np.array([0.005, -0.005]).reshape(-1, 1)  # initial value
dq = np.array([-0.1705, 0.1576]).reshape(-1, 1)
Weights = np.zeros((2 * nodes, 1)).reshape(-1, 1)
z = np.zeros(6).reshape(-1, 1)

Data_SS_Log = np.zeros((60001, 4))
Data_Tau_Log = np.zeros((Number_Major, 2))
qd3, qd4, dqd3, dqd4, ddqd3, ddqd4 = generate_ref()

# collect info
q_ref = np.concatenate((qd3, qd4), axis=1)
dq_ref = np.concatenate((dqd3, dqd4), axis=1)
total_eq = np.array([[], []])
total_edq = np.array([[], []])
total_q = np.array([[qd3[0, 0]], [qd4[0, 0]]])
total_dq = np.array([[dqd3[0, 0]], [dqd4[0, 0]]])
total_pred = np.array([[] for i in range(args.c_out)])

while count <= Number_All:  # 60001 or 60000?
    Data_SS_Log[count - 1, 0:2] = [q[0], q[1]]
    Data_SS_Log[count - 1, 2:4] = [dq[0], dq[1]]

    if (count - 1) % 5 == 0:  # MajorSteps # 控制器频率是实际的五分之一
        # update replay buffer
        if i > args.seq_len + 1:
            if args.input_type == 'actual':
                total_info = np.concatenate((total_q.transpose()[-args.seq_len - 1:-1, :], total_dq.transpose()[-args.seq_len - 1:-1, :]), axis=1)[:, :args.enc_in]
            total_label = np.concatenate((total_eq.transpose()[-args.pred_len:, :], total_edq.transpose()[-args.pred_len:, :]), axis=1)
            exps.update_buffer(total_info, total_label)
        if exps.get_buffer_size() >= args.minimal_size:
            exps.train_one_epoch()
            if args.input_type == 'actual':
                current_inputs = np.concatenate((total_q.transpose()[-args.seq_len:, :], total_dq.transpose()[-args.seq_len:, :]), axis=1)[:, :args.enc_in]
            # current_inputs = apply_norm(current_inputs)
            # [c_out, pred_len, 1]
            pred = exps.pred(current_inputs)[:, 0, :]
            pred = dynamic_adjust(pred, i, 5000)  # 大学习率和最开始输出控制
            e_pred = pred[:2, :]
            de_pred = pred[2:, :]
            # de_pred = dynamic_adjust(de_pred, counter, 5000)
            # de_pred = np.array([[0], [0]])
        else:
            e_pred = np.array([[0], [0]])
            de_pred = np.array([[0], [0]])
            pred = np.array([[0] for i in range(args.c_out)])

        q_sample = Encoder(q)
        Torque = Controller(Weights, e_pred[0, 0], e_pred[1, 0], qd3[i], qd4[i], dqd3[i], dqd4[i], ddqd3[i], ddqd4[i], q_sample, q, dq)
        # Torque = Controller(Weights, 0, 0, qd3[i], qd4[i], dqd3[i], dqd4[i], ddqd3[i], ddqd4[i], q_sample, q, dq)
        dq_OtherSteps = dq
        Weights = Dynamics_W(Weights, 0, 0, qd3[i], qd4[i], dqd3[i], dqd4[i], q_sample, dq, dt)
        ddq = Dynamics_ddq(Torque, q, dq, z)
        z = Dynamics_z(dq, z, dt)
        dq = Dynamics_dq(ddq, dq, dt)
        q = Dynamics_q(dq, q, dt)
        Data_Tau_Log[i, 0:2] = [Torque[2], Torque[3]]  # change
        #
        e_q_t = q - q_ref[i].reshape(-1, 1)
        e_dq_t = dq - dq_ref[i].reshape(-1, 1)
        total_eq = np.concatenate((total_eq, e_q_t), axis=1)
        total_edq = np.concatenate((total_edq, e_dq_t), axis=1)
        total_dq = np.concatenate((total_dq, dq), axis=1)
        total_q = np.concatenate((total_q, q), axis=1)
        total_pred = np.concatenate((total_pred, pred), axis=1)
        i += 1  # i的终止值是12001
    else:
        Weights = Dynamics_W(Weights, 0, 0, qd3[i - 1], qd4[i - 1], dqd3[i - 1], dqd4[i - 1], q_sample, dq_OtherSteps, dt)
        ddq = Dynamics_ddq(Torque, q, dq, z)
        z = Dynamics_z(dq, z, dt)
        dq = Dynamics_dq(ddq, dq, dt)
        q = Dynamics_q(dq, q, dt)

    T += dt  # 实际时间
    count += 1
    if count % 300 == 0:
        print(count)
        print(e_pred)

all_pred = total_pred.transpose()[args.seq_len:, :]  # 去除掉最开始没有介入网络的部分
all_true = np.concatenate((total_eq, total_edq), axis=0).transpose()[args.seq_len:, :]

visual_all(all_true, all_pred, folder_path + 'loss_.png', args.c_out, show_plot=args.show_plot)
# visual plot
q3_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 0]
q4_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 1]
dq3_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 2]
dq4_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 3]
all_act = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), :]
# record data
eq_rms = np.sqrt(np.mean((q_ref - all_act[:, 0:2]) ** 2))
edq_rms = np.sqrt(np.mean((dq_ref - all_act[:, 2:]) ** 2))
ep_rms = np.sqrt(np.mean((all_pred - all_true) ** 2))
eq_target = 0.0002598054130803498
edq_tarrget = 0.002690652238554081
info = 'eq{}_edq{}_ep{}_'.format(eq_target-eq_rms, edq_tarrget- edq_rms, ep_rms)
write_info("./results/result_rbf.txt", setting, info)



value_l = [qd3, qd4, dqd3, dqd4, Data_SS_Log, Data_Tau_Log, Number_Major, T_final, dt, q3_12001, q4_12001, dq3_12001, dq4_12001]
key_l = ['qd3', 'qd4', 'dqd3', 'dqd4', 'Data_SS_Log', 'Data_Tau_Log', 'Number_Major', 'T_final', 'dt', 'q3_12001', 'q4_12001', 'dq3_12001', 'dq4_12001']
generate_mat('baseline/basic_rbf', value_l, key_l)
plot_data(qd3, qd4, dqd3, dqd4, Data_SS_Log, Data_Tau_Log, Number_Major, T_final, dt, folder_path=folder_path, show_other=True, show_plot=args.show_plot)
