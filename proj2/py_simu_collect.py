import numpy as np
from test_model.all_func import stiffness, coriolis, gravity, dynamics, disturbance, plot_all, arrays_to_dataframe
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.integrate import odeint



# sys config
folder_path = './results/'
counter = 0
sample_rate = 0.0025
total_t = 32
total_epoch = int(total_t/sample_rate)
dim =2
q_init = np.array([[- np.pi / 2], [1.20919276]]) # -1,0
dq_init = np.array([[0.78539816], [-0.00390299]])
#dq_init = np.array([[0], [0]])
t1 = np.arange(0, total_t+sample_rate, sample_rate)
t1 = t1.reshape(-1, 1)
tout = t1
period = 4
q1_ref = np.sin(np.pi / period * tout) - np.pi / 2
q2_ref = np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2))
dq1_ref = np.pi / period * np.cos(np.pi / period * tout)
dq2_ref = np.pi / 2 * -np.pi / 8 * np.sin(np.pi / 4 * (tout + np.pi / 2)) * np.sin(
            np.pi / 8 * (tout + np.pi / 2)) + np.pi / 2 * np.pi / 4 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.cos(
            np.pi / 4 * (tout + np.pi / 2))
q_ref = np.concatenate((q1_ref, q2_ref), axis=1)
dq_ref = np.concatenate((dq1_ref, dq2_ref), axis=1)
e_predv = np.array([[0], [0]])
total_eq = np.array([[], []])
total_edq = np.array([[], []])
total_u = np.array([[0], [0]])
total_q = q_init
total_dq = dq_init
#pd controller
kp1 = 1000
kp2 = 800
kd1=500
kd2=80
kp = np.diag([kp1, kp2])
kd = np.diag([kd1, kd2])
# load historical error
df = pd.read_pickle('data/pd_train.pkl')
e_preds = df[['eq1', 'eq2']].values.transpose()


while counter < total_epoch:
    if counter <= 128001: # 21 128001时候就是单纯的collect
        e_pred = np.array([[0], [0]])

    e_q_t = q_init - q_ref[counter].reshape(dim, 1)
    e_q = e_q_t + 1*e_pred
    e_pred = e_q_t
    e_dq = dq_init - dq_ref[counter].reshape(dim, 1)
    u_pd = -kp@e_q - kd@e_dq
    u_pd = np.clip(u_pd, -130, 130)
    M = stiffness(q_init)
    C = coriolis(q_init, dq_init)
    G = gravity(q_init)
    disturb = disturbance(counter*sample_rate)
    TL = u_pd - C - G + disturb
    ddq = dynamics(TL, M)
    dq = dq_init + ddq*sample_rate
    q = q_init + dq*sample_rate
    dq_init = dq
    q_init = q
    counter += 1
    total_edq = np.concatenate((total_edq, e_dq), axis=1)
    total_eq = np.concatenate((total_eq, e_q), axis=1)
    total_dq = np.concatenate((total_dq, dq), axis=1)
    total_q = np.concatenate((total_q, q), axis=1)
    total_u = np.concatenate((total_u, u_pd), axis=1)
    print(counter)
    # 这样相当于是有重力但是没有disturbance

total_q = total_q.transpose()   #(12800,2) ,去除最后一个值保持和e维度一直
total_dq = total_dq.transpose()
total_eq = total_eq.transpose()
total_edq = total_edq.transpose()
sTau = total_u.transpose()
# df = arrays_to_dataframe(total_q, total_dq, total_eq, total_edq)
df = arrays_to_dataframe(q_ref[:-1, :], dq_ref[:-1, :], total_eq, total_edq)
df.to_pickle('data/pd_ref_train.pkl')
q = total_q
dq = total_dq
tout = np.linspace(0, 32, num=len(t1)) #不使用np速度回很慢
plot_all(q, dq, sTau, tout, q_ref, dq_ref, folder_path)
