import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
import math
import scipy.io as sio


def generate_mat(folder_path, value_l, label_l):
    result_dict = {}
    for label, value in zip(label_l, value_l):
        result_dict[label] = value
    sio.savemat(folder_path + 'data.mat', result_dict)


def apply_norm(data):
    mean = data.mean(0)
    std = data.std(0)
    return (data - mean) / std


def stiffness(q):
    q1, q2 = q[0, 0], q[1, 0]

    s1, s2 = np.sin(q1), np.sin(q2)
    c1, c2 = np.cos(q1), np.cos(q2)

    m1, m2 = 5, 5
    l1, l2 = 1, 1

    g = 9.81

    M11 = m1 * l1 ** 2 / 3 + m2 * l1 ** 2 + m2 * l2 ** 2 / 4 + m2 * l1 * l2 * c2
    M12 = m2 * l2 ** 2 / 4 + m2 * l1 * l2 * c2 / 2
    M22 = m2 * l2 ** 2 / 3

    M = np.array([[M11], [M12],
                  [M12], [M22]])

    return M


def coriolis(q, dq):
    q1, q2 = q[0, 0], q[1, 0]
    dq1, dq2 = dq[0, 0], dq[1, 0]

    s1, s2 = np.sin(q1), np.sin(q2)
    c1, c2 = np.cos(q1), np.cos(q2)

    m1, m2 = 5, 5
    l1, l2 = 1, 1

    g = 9.80665

    C11 = -m2 * l1 * l2 * s2 * dq2
    C12 = -m2 * l1 * l2 * s2 * dq2 / 2
    C21 = m2 * l1 * l2 * s2 * dq1 / 2
    C22 = 0

    CM = np.array([[C11, C12],
                   [C21, C22]])

    C = np.dot(CM, dq)

    return C


def gravity(q):
    q1, q2 = q[0, 0], q[1, 0]

    s1, s2 = np.sin(q1), np.sin(q2)
    c1, c2 = np.cos(q1), np.cos(q2)
    c12 = np.cos(q1 + q2)

    m1, m2 = 5, 5
    l1, l2 = 1, 1

    g = 9.80665

    G1 = m1 * g * l1 * c1 / 2 + m2 * g * l1 * c1 + m2 * g * l2 * c12 / 2
    G2 = m2 * g * l2 * c12 / 2

    G = np.array([[G1], [G2]])

    return G


# TL[2,1] M[2,2]
def dynamics(TL, M):
    M = M.reshape(-1)
    # print(M.shape)
    M22 = np.array([[M[0], M[1]],
                    [M[2], M[3]]])
    # print(M22.shape)
    ddq = np.linalg.solve(M22, TL)

    return ddq


def disturbance(t):
    if t > 10:
        d = 50.0 * np.array([np.sin(0.6 * (t - 10)), np.sin(0.6 * (t - 10))])
    else:
        d = np.zeros(2)
    return d.reshape(2, 1)


def plot_all(q, dq, sTau, tout, q_ref, dq_ref, folder_path, show_other=True, flag='bs'):
    # load baseline data
    if flag == 'bs':
        baseline_f = 'baseline/basic_sine/'
    elif flag == 'ss':
        baseline_f = 'baseline/shake_sine/'
    # print(baseline_f)
    if show_other:
        mat_g = scipy.io.loadmat(baseline_f + 'gravity.mat')
        mat_og = scipy.io.loadmat(baseline_f + 'gravity_offset.mat')
        all_mat = [mat_g, mat_og]  # [mat_g, mat_og]
        labels = ['g', 'og']

    counter = 0
    plt.figure(1)
    plt.subplot(321)
    plt.plot(tout, q_ref[:, 0], linewidth=0.5, label='gt')
    plt.plot(tout, q[:, 0], linewidth=0.5, label='ours')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, all_mat[i]['q'][:, 0], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$q_1$ (rad)')

    plt.subplot(323)
    plt.plot(tout, abs(q_ref[:, 0] - q[:, 0]) * 180 / np.pi, linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, abs(q_ref[:, 0] - all_mat[i]['q'][:, 0]) * 180 / np.pi, linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e_{q_1}$ (deg)')

    plt.subplot(322)
    plt.plot(tout, q_ref[:, 1], linewidth=0.5, label='gt')
    plt.plot(tout, q[:, 1], linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, all_mat[i]['q'][:, 1], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$q_2$ (rad)')

    plt.subplot(324)
    plt.plot(tout, np.abs(q_ref[:, 1] - q[:, 1]) * 180 / np.pi, linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, np.abs(q_ref[:, 1] - all_mat[i]['q'][:, 1]) * 180 / np.pi, linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e_{q_2}$ (deg)')

    plt.subplot(325)
    plt.plot(tout, sTau[:, 0], linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, all_mat[i]['sTau'][:, 0], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$\tau_1$ [Nm]')

    plt.subplot(326)
    plt.plot(tout, sTau[:, 1], linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, all_mat[i]['sTau'][:, 1], linewidth=0.5, label=labels[i])
    plt.ylabel(r'$\tau_2$ [Nm]')
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(folder_path + 'pos_{}.svg'.format(counter), format='svg')

    # plot the second graph
    plt.figure(2)
    plt.subplot(221)
    t = tout
    pi = np.pi
    plt.plot(tout, dq_ref[:, 0], linewidth=0.5, label='gt')
    plt.plot(tout, dq[:, 0], linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, all_mat[i]['dq'][:, 0], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$\dot{q}_1$ (rad/s)')

    plt.subplot(223)
    plt.plot(tout, np.abs(dq_ref[:, 0] - dq[:, 0]), linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, np.abs(dq_ref[:, 0] - all_mat[i]['dq'][:, 0]), linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e\dot{q}_1$ (rad/s)')

    plt.subplot(222)
    plt.plot(tout, dq_ref[:, 1], linewidth=0.5, label='gt')
    plt.plot(tout, dq[:, 1], linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, all_mat[i]['dq'][:, 1], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$\dot{q}_2$ (rad/s)')

    plt.subplot(224)
    plt.plot(tout, np.abs(dq_ref[:, 1] - dq[:, 1]), linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(tout, np.abs(dq_ref[:, 1] - all_mat[i]['dq'][:, 1]), linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.ylabel(r'$e\dot{q}_2$ (rad/s)')
    plt.savefig(folder_path + 'vel_{}.svg'.format(counter), format='svg')
    plt.show()


def arrays_to_dataframe(total_q, total_dq, total_eq, total_edq):
    """
    Convert four numpy arrays with shape (12800, 2) into a DataFrame with eight columns.

    Parameters:
    - total_q, total_dq, total_eq, total_edq: numpy.array of shape (12800, 2)

    Returns:
    - DataFrame with columns 'q1', 'q2', 'dq1', 'dq2', 'eq1', 'eq2', 'edq1', 'edq2'
    """

    # Split each numpy.array into separate columns
    q1, q2 = total_q[:, 0], total_q[:, 1]
    dq1, dq2 = total_dq[:, 0], total_dq[:, 1]
    eq1, eq2 = total_eq[:, 0], total_eq[:, 1]
    edq1, edq2 = total_edq[:, 0], total_edq[:, 1]

    # Create and return a DataFrame
    return pd.DataFrame({
        'q1': q1,
        'q2': q2,
        'dq1': dq1,
        'dq2': dq2,
        'eq1': eq1,
        'eq2': eq2,
        'edq1': edq1,
        'edq2': edq2
    })


def write_info(folder_p, setting, info):
    f = open(folder_p, 'a')
    f.write(setting + "  \n")
    f.write(info)
    f.write('\n')
    f.write('\n')
    f.close()


def visual_all(res_true, res_pred, name, c_out, show_plot=1):
    fig, axs = plt.subplots(2, math.ceil(c_out / 2), figsize=(12, 6))
    idx = 0
    # 这里有个bug就是，如果列为1，那么就会导致没有len(axs[0])
    if math.ceil(c_out / 2) > 1:
        for i in range(len(axs)):
            for j in range(len(axs[0])):
                if idx >= c_out:
                    # 处理奇数张图
                    break
                # 绘制第一个子图
                axs[i][j].plot(res_true[:, idx], label='GroundTruth')
                axs[i][j].plot(res_pred[:, idx], label='Prediction')
                axs[i][j].legend()
                axs[i][j].set_title('Subplot {}'.format(idx))
                idx += 1
    else:
        for i in range(len(axs)):
            if idx >= c_out:
                # 处理奇数张图
                break
            # 绘制第一个子图
            axs[i].plot(res_true[:, idx], label='GroundTruth')
            axs[i].plot(res_pred[:, idx], label='Prediction')
            axs[i].legend()
            axs[i].set_title('Subplot {}'.format(idx))
            idx += 1

    plt.savefig(name)
    if show_plot:
        plt.show()


def dynamic_adjust(pred, current_epoch, threshold):
    if current_epoch < threshold:
        return pred * current_epoch / threshold
    else:
        return pred


def generate_ref(flag, tout):
    # basic sine function
    if flag == 'bs':
        q1_ref = np.sin(np.pi / 4 * tout) - np.pi / 2
        q2_ref = np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2))
        dq1_ref = np.pi / 4 * np.cos(np.pi / 4 * tout)
        dq2_ref = np.pi / 2 * -np.pi / 8 * np.sin(np.pi / 4 * (tout + np.pi / 2)) * np.sin(np.pi / 8 * (tout + np.pi / 2)) + np.pi / 2 * np.pi / 4 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.cos(
            np.pi / 4 * (tout + np.pi / 2))
    # shake sine function
    if flag == 'ss':
        Tf = int(tout[-1])
        w = np.pi / 3
        A = 1 / Tf * tout + 1
        omega = w + w / Tf * tout
        q1_ref = A * np.sin(omega * tout) - np.pi / 2
        q2_ref = np.pi / 2 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.sin(np.pi / 4 * (tout + np.pi / 2))
        dq1_ref = 1 / Tf * np.sin(omega * tout) + A * np.cos(omega * tout) * (2 * w / Tf * tout + w)
        dq2_ref = np.pi / 2 * -np.pi / 8 * np.sin(np.pi / 4 * (tout + np.pi / 2)) * np.sin(np.pi / 8 * (tout + np.pi / 2)) + np.pi / 2 * np.pi / 4 * np.cos(np.pi / 8 * (tout + np.pi / 2)) * np.cos(
            np.pi / 4 * (tout + np.pi / 2))
    q_ref = np.concatenate((q1_ref, q2_ref), axis=1)
    dq_ref = np.concatenate((dq1_ref, dq2_ref), axis=1)
    return q_ref, dq_ref
