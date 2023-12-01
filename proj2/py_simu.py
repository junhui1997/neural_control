import numpy as np
from test_model.all_func import stiffness, coriolis, gravity, dynamics, disturbance, plot_all, generate_ref
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint



# sys config
task_name = 'ss'
folder_path = './results/'
counter = 0
sample_rate = 0.0025
total_t = 32
total_epoch = int(total_t/sample_rate)
dim =2
q_init = np.array([[-1], [0]])
dq_init = np.array([[0], [0]])
t1 = np.arange(0, 32.0025, 0.0025)
t1 = t1.reshape(-1, 1)
tout = t1
q_ref, dq_ref = generate_ref(task_name, tout)
# total_eq = np.array([[], []])
# total_edq = np.array([[], []])
total_q = q_init
total_dq = dq_init
total_u = np.array([[0], [0]])
#pd controller
kp1 = 1000
kp2 = 800
kd1=500
kd2=80
kp = np.diag([kp1, kp2])
kd = np.diag([kd1, kd2])
while counter < total_epoch:
    e_q = (q_init - q_ref[counter].reshape(dim, 1))
    e_dq = (dq_init - dq_ref[counter].reshape(dim, 1))
    u_pd = -kp@e_q - kd@e_dq
    #u_pd = np.clip(u_pd, -130, 130) # 把这行注释了，两边作图完全一致
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
    total_dq = np.concatenate((total_dq, dq), axis=1)
    total_q = np.concatenate((total_q, q), axis=1)
    total_u = np.concatenate((total_u, u_pd), axis=1)
    print(counter)
    # 这样相当于是有重力但是没有disturbance

q = total_q.transpose()
dq = total_dq.transpose()
sTau = total_u.transpose()
tout = np.linspace(0, 32, num=len(t1)) #不使用np速度回很慢
plot_all(q, dq, sTau, tout, q_ref, dq_ref, folder_path, flag=task_name)
