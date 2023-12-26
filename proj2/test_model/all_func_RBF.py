import numpy as np
import scipy
import matplotlib.pyplot as plt


def generate_ref(flag='basic'):
    # basic sine function
    if flag == 'basic':
        data = scipy.io.loadmat('test_model/ref_data/MAT_Input_Ref_DesiredJointsTrajForSpiralCircle_ClosedLoopCode.mat')
        qd3 = data['qd3']
        qd4 = data['qd4']
        dqd3 = data['dqd3']
        dqd4 = data['dqd4']
        ddqd3 = data['ddqd3']
        ddqd4 = data['ddqd4']

    return qd3, qd4, dqd3, dqd4, ddqd3, ddqd4


def generate_info():
    return None


def plot_data(qd3, qd4, dqd3, dqd4, Data_SS_Log, Data_Tau_Log, Number_Major, T_final, dt, folder_path, show_other=True, flag='br', show_plot=1):
    t1 = np.arange(0, T_final + dt, dt)
    t2 = np.arange(0, T_final + 5 * dt, 5 * dt)
    if flag == 'br':
        baseline_f = 'baseline/basic_rbf/'
    if show_other:
        mat_b = scipy.io.loadmat(baseline_f + 'basic.mat')
        mat_offset = scipy.io.loadmat(baseline_f + 'matlab_offset.mat')
        all_mat = [mat_b]  # [mat_b, mat_offset]
        labels = ['basic']  # ['basic', 'offset']


    plt.figure(1, figsize=(12, 8))

    plt.subplot(221)
    plt.plot(t2, qd3, 'k', linewidth=0.5, label='ref')
    plt.plot(t1, Data_SS_Log[:, 0], 'g--', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t1, all_mat[i]['Data_SS_Log'][:, 0], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(222)
    plt.plot(t2, qd4, 'k', linewidth=0.5, label='ref')
    plt.plot(t1, Data_SS_Log[:, 1], 'g--', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t1, all_mat[i]['Data_SS_Log'][:, 1], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(223)
    plt.plot(t2, dqd3, 'k', linewidth=0.5, label='ref')
    plt.plot(t1, Data_SS_Log[:, 2], 'g--', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t1, all_mat[i]['Data_SS_Log'][:, 2], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(224)
    plt.plot(t2, dqd4, 'k', linewidth=0.5, label='ref')
    plt.plot(t1, Data_SS_Log[:, 3], 'g--', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t1, all_mat[i]['Data_SS_Log'][:, 3], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(folder_path + '1_.svg', format='svg')

    q3_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 0]
    q4_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 1]
    dq3_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 2]
    dq4_12001 = Data_SS_Log[np.arange(0, 5 * Number_Major, 5), 3]

    plt.figure(2, figsize=(12, 8))

    plt.subplot(221)
    # a = np.random.rand(12000, 1)
    # b = np.random.rand(12000)
    # c = a - b
    # c.shape c是（12000,12000），这里会导致数据爆炸
    plt.plot(t2, np.abs(qd3.reshape(-1) - q3_12001) * 180 / np.pi, 'r', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t2, np.abs(qd3.reshape(-1) - all_mat[i]['q3_12001'].reshape(-1)) * 180 / np.pi, linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(222)
    plt.plot(t2, np.abs(qd4.reshape(-1) - q4_12001) * 180 / np.pi, 'r', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t2, np.abs(qd4.reshape(-1) - all_mat[i]['q4_12001'].reshape(-1)) * 180 / np.pi, linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(223)
    plt.plot(t2, np.abs(dqd3.reshape(-1) - dq3_12001), 'r', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t2, np.abs(dqd3.reshape(-1) - all_mat[i]['dq3_12001'].reshape(-1)), linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(224)
    plt.plot(t2, np.abs(dqd4.reshape(-1) - dq4_12001), 'r', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t2, np.abs(dqd4.reshape(-1) - all_mat[i]['dq4_12001'].reshape(-1)), linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(folder_path + '2_.svg', format='svg')

    plt.figure(3, figsize=(12, 8))
    plt.subplot(221)
    plt.plot(t2, Data_Tau_Log[:, 0], 'r', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t2, all_mat[i]['Data_Tau_Log'][:, 0], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(222)
    plt.plot(t2, Data_Tau_Log[:, 1], 'r', linewidth=0.5, label='our')
    if show_other:
        for i in range(len(all_mat)):
            plt.plot(t2, all_mat[i]['Data_Tau_Log'][:, 1], linewidth=0.5, label=labels[i])
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(folder_path + '3_.svg', format='svg')
    if show_plot:
        plt.show()
    plt.clf()


def Dynamics_ddq(Torque, q, dq, z):
    q3 = q[0, 0]
    q4 = q[1, 0]
    dq3 = dq[0, 0]
    dq4 = dq[1, 0]

    M11 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 - (9666515293019412421 * np.sin(2 * q3 + q4)) / 147573952589676412928 - (7347855351585683 * np.cos(2 * q3)) / 18014398509481984 - (
            36945533070432904259 * np.cos(2 * q3 + 2 * q4)) / 2361183241434822606848 - (2242006792538445303 * np.sin(2 * q3 + 2 * q4)) / 1208925819614629174706176 - (
                  3932438768524053 * np.cos(2 * q3 + q4)) / 590295810358705651712 + (9666515293019412421 * np.sin(q4)) / 147573952589676412928 + 6993021903434165153791 / 9444732965739290427392
    M12 = (1566193994333460609165 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264027 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272 - (
            158738597992664517 * np.sin(q3 + np.pi / 2)) / 1152921504606846976 - 23750155206264607 / 144115188075855872
    M13 = (1566193994333460609165 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264027 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272 - (
            158738597992664517 * np.sin(q3 + np.pi / 2)) / 1152921504606846976
    M14 = (1566193994333460609165 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264027 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272
    M15 = -(3951197109736239241155 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (2242006792538445279 * np.sin(q3 + q4 + np.pi / 2)) / 1208925819614629174706176 - (
            3932438768524053 * np.cos(q3 + np.pi / 2)) / 590295810358705651712 - 2364964252341131 / 5316911983139663491615228241121378304
    M16 = (392149993849384621 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264035 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272
    M22 = (6599725983735443 * np.cos(q3)) / 4503599627370496 + (3932438768524053 * np.cos(q4)) / 295147905179352825856 + (9666515293019412421 * np.sin(q4)) / 73786976294838206464 + (
            8575983760191473 * np.cos(q3) * np.cos(q4)) / 590295810358705651712 + (21081034709077362555 * np.cos(q3) * np.sin(q4)) / 147573952589676412928 + (
                  21081034709077362555 * np.cos(q4) * np.sin(q3)) / 147573952589676412928 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 590295810358705651712 + 802526512685661487937 / 295147905179352825856
    M23 = (6599725983735443 * np.cos(q3)) / 9007199254740992 + (3932438768524053 * np.cos(q4)) / 295147905179352825856 + (9666515293019412421 * np.sin(q4)) / 73786976294838206464 + (
            8575983760191473 * np.cos(q3) * np.cos(q4)) / 1180591620717411303424 + (21081034709077362555 * np.cos(q3) * np.sin(q4)) / 295147905179352825856 + (
                  21081034709077362555 * np.cos(q4) * np.sin(q3)) / 295147905179352825856 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 1180591620717411303424 + 292870962545184539457 / 295147905179352825856
    M24 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (9666515293019412421 * np.sin(q4)) / 147573952589676412928 + (
            8575983760191473 * np.cos(q3) * np.cos(q4)) / 1180591620717411303424 + (21081034709077362555 * np.cos(q3) * np.sin(q4)) / 295147905179352825856 + (
                  21081034709077362555 * np.cos(q4) * np.sin(q3)) / 295147905179352825856 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 1180591620717411303424 + 36296309188291035969 / 295147905179352825856
    M25 = (2230692869490939 * np.sin(q4)) / 1152921504606846976 + (304047738388393 * np.cos(q3) * np.sin(q4)) / 144115188075855872 + (
            304047738388393 * np.cos(q4) * np.sin(q3)) / 144115188075855872 + 315508820003156273 / 2361183241434822606848
    M26 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (2962264730236869 * np.sin(q4)) / 147573952589676412928 + (8575983760191473 * np.cos(q3) * np.cos(q4)) / 1180591620717411303424 + (
            6460198293038971 * np.cos(q3) * np.sin(q4)) / 295147905179352825856 + (6460198293038971 * np.cos(q4) * np.sin(q3)) / 295147905179352825856 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 1180591620717411303424 + 29720514162674078529 / 590295810358705651712
    M33 = (3932438768524053 * np.cos(q4)) / 295147905179352825856 + (9666515293019412421 * np.sin(q4)) / 73786976294838206464 + 292870962545184539457 / 295147905179352825856
    M34 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (9666515293019412421 * np.sin(q4)) / 147573952589676412928 + 36296309188291035969 / 295147905179352825856
    M35 = (2230692869490939 * np.sin(q4)) / 1152921504606846976 + 315508820003156273 / 2361183241434822606848
    M36 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (2962264730236869 * np.sin(q4)) / 147573952589676412928 + 29720514162674078529 / 590295810358705651712
    M44 = 0.1230
    M45 = 1.3362e-04
    M46 = 0.0503
    M55 = 0.0529
    M56 = 1.2030e-06
    M66 = 0.0503

    M = np.array([M11, M12, M13, M14, M15, M16,
                  M12, M22, M23, M24, M25, M26,
                  M13, M23, M33, M34, M35, M36,
                  M14, M24, M34, M44, M45, M46,
                  M15, M25, M35, M45, M55, M56,
                  M16, M26, M36, M46, M56, M66]).reshape(6, 6)

    # C
    C1 = (158738597992664517 * dq3 ** 2 * np.sin(q3)) / 1152921504606846976 - \
         (788574876594818524558499406425187 * dq3 ** 2 * np.cos(q3)) / 93536104789177786765035829293842113257979682750464 - \
         (31766188111850551462598880893467128091902779047347567 * dq3 ** 2 * np.cos(q3) * np.cos(q4)) / 1532495540865888858358347027150309183618739122183602176 - \
         (31766188111850551462598880893467128091902779047347567 * dq4 ** 2 * np.cos(q3) * np.cos(q4)) / 1532495540865888858358347027150309183618739122183602176 + \
         (329958928438908233217359294665525876989291241165 * dq3 ** 2 * np.cos(q3) * np.sin(q4)) / 191561942608236107294793378393788647952342390272950272 + \
         (329958928438908233217359294665525876989291241165 * dq3 ** 2 * np.cos(q4) * np.sin(q3)) / 191561942608236107294793378393788647952342390272950272 + \
         (329958928438908233217359294665525876989291241165 * dq4 ** 2 * np.cos(q3) * np.sin(q4)) / 191561942608236107294793378393788647952342390272950272 + \
         (329958928438908233217359294665525876989291241165 * dq4 ** 2 * np.cos(q4) * np.sin(q3)) / 191561942608236107294793378393788647952342390272950272 + \
         (31766188111850551462598880893467128091902779047347567 * dq3 ** 2 * np.sin(q3) * np.sin(q4)) / 1532495540865888858358347027150309183618739122183602176 + \
         (31766188111850551462598880893467128091902779047347567 * dq4 ** 2 * np.sin(q3) * np.sin(q4)) / 1532495540865888858358347027150309183618739122183602176 - \
         (31766188111850551296445381420352643978926896512304495 * dq3 * dq4 * np.cos(q3) * np.cos(q4)) / 766247770432944429179173513575154591809369561091801088 + \
         (329958928438908233217359294666797622934896747981 * dq3 * dq4 * np.cos(q3) * np.sin(q4)) / 95780971304118053647396689196894323976171195136475136 + \
         (329958928438908233217359294666797622934896747981 * dq3 * dq4 * np.cos(q4) * np.sin(q3)) / 95780971304118053647396689196894323976171195136475136 + \
         (31766188111850551296445381420352643978926896512304495 * dq3 * dq4 * np.sin(q3) * np.sin(q4)) / 766247770432944429179173513575154591809369561091801088

    C2 = (9666515293019412421 * dq4 ** 2 * np.cos(q4)) / 147573952589676412928 - \
         (3932438768524053 * dq4 ** 2 * np.sin(q4)) / 590295810358705651712 - \
         (6599725983735443 * dq3 ** 2 * np.sin(q3)) / 9007199254740992 - \
         (3932438768524053 * dq3 * dq4 * np.sin(q4)) / 295147905179352825856 + \
         (21081034709077362555 * dq3 ** 2 * np.cos(q3) * np.cos(q4)) / 295147905179352825856 + \
         (21081034709077362555 * dq4 ** 2 * np.cos(q3) * np.cos(q4)) / 295147905179352825856 - \
         (8575983760191473 * dq3 ** 2 * np.cos(q3) * np.sin(q4)) / 1180591620717411303424 - \
         (8575983760191473 * dq3 ** 2 * np.cos(q4) * np.sin(q3)) / 1180591620717411303424 - \
         (8575983760191473 * dq4 ** 2 * np.cos(q3) * np.sin(q4)) / 1180591620717411303424 - \
         (8575983760191473 * dq4 ** 2 * np.cos(q4) * np.sin(q3)) / 1180591620717411303424 - \
         (21081034709077362555 * dq3 ** 2 * np.sin(q3) * np.sin(q4)) / 295147905179352825856 - \
         (21081034709077362555 * dq4 ** 2 * np.sin(q3) * np.sin(q4)) / 295147905179352825856 + \
         (9666515293019412421 * dq3 * dq4 * np.cos(q4)) / 73786976294838206464 + \
         (21081034709077362555 * dq3 * dq4 * np.cos(q3) * np.cos(q4)) / 147573952589676412928 - \
         (8575983760191473 * dq3 * dq4 * np.cos(q3) * np.sin(q4)) / 590295810358705651712 - \
         (8575983760191473 * dq3 * dq4 * np.cos(q4) * np.sin(q3)) / 590295810358705651712 - \
         (21081034709077362555 * dq3 * dq4 * np.sin(q3) * np.sin(q4)) / 147573952589676412928

    C3 = (9666515293019412421 * dq4 ** 2 * np.cos(q4)) / 147573952589676412928 - \
         (3932438768524053 * dq4 ** 2 * np.sin(q4)) / 590295810358705651712 - \
         (3932438768524053 * dq3 * dq4 * np.sin(q4)) / 295147905179352825856 + \
         (9666515293019412421 * dq3 * dq4 * np.cos(q4)) / 73786976294838206464

    C4 = (3932438768524053 * dq3 ** 2 * np.sin(q4)) / 590295810358705651712 - \
         (9666515293019412421 * dq3 ** 2 * np.cos(q4)) / 147573952589676412928

    C5 = -(7541740619071967 * dq3 * dq4) / 9444732965739290427392 - \
         (7541740619071967 * dq3 ** 2) / 18889465931478580854784 - \
         (7541740619071967 * dq4 ** 2) / 18889465931478580854784 - \
         (2230692869490939 * dq3 ** 2 * np.cos(q4)) / 1152921504606846976

    C6 = (3932438768524053 * dq3 ** 2 * np.sin(q4)) / 590295810358705651712 + \
         (4369139136790775 * dq3 * dq4) / 1180591620717411303424 + \
         (4369139136790775 * dq3 ** 2) / 2361183241434822606848 + \
         (4369139136790775 * dq4 ** 2) / 2361183241434822606848 - \
         (2962264730236869 * dq3 ** 2 * np.cos(q4)) / 147573952589676412928

    C = np.array([C1, C2, C3, C4, C5, C6]).reshape(-1, 1)

    # G
    G1 = 0
    G2 = (817603655937935 * np.cos(q3 + q4 + np.pi / 2)) / 4611686018427387904 + \
         (10048952709718828759483 * np.sin(q3 + q4 + np.pi / 2)) / 5764607523034234880000 + \
         (12583885985274322167 * np.cos(q3 + np.pi / 2)) / 703687441776640000 + \
         10874744537588643 / 5070602400912917605986812821504

    G3 = (12583885985274322167 * np.cos(q3 + np.pi / 2)) / 703687441776640000 + \
         (817603655937935 * np.cos(q4) * np.cos(q3 + np.pi / 2)) / 4611686018427387904 + \
         (10048952709718828759483 * np.cos(q4) * np.sin(q3 + np.pi / 2)) / 5764607523034234880000 + \
         (10048952709718828759483 * np.cos(q3 + np.pi / 2) * np.sin(q4)) / 5764607523034234880000 - \
         (817603655937935 * np.sin(q4) * np.sin(q3 + np.pi / 2)) / 4611686018427387904

    G4 = (817603655937935 * np.cos(q4) * np.cos(q3 + np.pi / 2)) / 4611686018427387904 + \
         (10048952709718828759483 * np.cos(q4) * np.sin(q3 + np.pi / 2)) / 5764607523034234880000 + \
         (10048952709718828759483 * np.cos(q3 + np.pi / 2) * np.sin(q4)) / 5764607523034234880000 - \
         (817603655937935 * np.sin(q4) * np.sin(q3 + np.pi / 2)) / 4611686018427387904

    G5 = (11873003616788183147 * np.sin(q3 + q4 + np.pi / 2)) / 230584300921369395200
    G6 = (817603655937935 * np.cos(q3 + q4 + np.pi / 2)) / 4611686018427387904 + \
         (4927137821310187 * np.sin(q3 + q4 + np.pi / 2)) / 9223372036854775808

    G = np.array([G1, G2, G3, G4, G5, G6]).reshape(-1, 1)

    # 4F 把两个F合在一起了
    z3 = z[2, 0]
    z4 = z[3, 0]
    dz3 = (32880603970571302491146779030281 * dq3 * np.arctan(1000 * dq3) ** 2) / 81129638414606681695789005144064 - \
          (13366329615528018129 * dq3 * z3 * np.arctan(1000 * dq3)) / (1801439850948198400 *
                                                                       ((6177572351796469 * np.arctan(
                                                                           (5734161139222659 * dq3 * np.arctan(1000 * dq3)) / 90071992547409920)) / 144115188075855872 + 1003 / 2500))

    dz4 = (32880603970571302491146779030281 * dq4 * np.arctan(1000 * dq4) ** 2) / 81129638414606681695789005144064 - \
          (13366329615528018129 * dq4 * z4 * np.arctan(1000 * dq4)) / (1801439850948198400 *
                                                                       ((6177572351796469 * np.arctan(
                                                                           (5734161139222659 * dq4 * np.arctan(1000 * dq4)) / 90071992547409920)) / 144115188075855872 + 1003 / 2500))

    F1 = 0
    F2 = 0
    F3 = (797302866510865 * dq3) / 288230376151711744 + (14749 * dz3) / 2500 + (2331 * z3) / 200 - \
         (20755908149554467968595694231197 * dq3 ** 2 * np.arctan(1000 * dq3)) / 21267647932558653966460912964485513216

    F4 = (797302866510865 * dq4) / 288230376151711744 + (14749 * dz4) / 2500 + (2331 * z4) / 200 - \
         (20755908149554467968595694231197 * dq4 ** 2 * np.arctan(1000 * dq4)) / 21267647932558653966460912964485513216

    F4 = F4 / 5
    F5 = 0
    F6 = 0

    F = np.array([F1, F2, F3, F4, F5, F6]).reshape(-1, 1)

    #  5 ddq
    ddq = np.linalg.solve(M, (Torque - C - G - F))
    # to modify
    ddq_max = np.pi
    ddq = np.clip(ddq, -ddq_max, ddq_max)
    ddq_out = np.array([ddq[2][0], ddq[3][0]]).reshape(-1, 1)

    # to check
    # to_check = scipy.io.loadmat('baseline/to_check/ddq.mat')
    # m_check = to_check['M_66']
    # c_check = to_check['C']
    # f_check = to_check['F']
    # g_check = to_check['G']
    # res1 = np.sum(M-m_check)
    # res2 = np.sum(C-c_check)
    # res3 = np.sum(F-f_check)
    # res4 = np.sum(G-g_check)

    return ddq_out


# 鬃毛变量求解
def Dynamics_z(dq, z, dt):
    z3 = z[2, 0]
    z4 = z[3, 0]
    dq3 = dq[0, 0]
    dq4 = dq[1, 0]

    dz1 = 0
    dz2 = 0
    dz3 = (
            (32880603970571302491146779030281 * dq3 * np.arctan(1000 * dq3) ** 2) /
            81129638414606681695789005144064 -
            (13366329615528018129 * dq3 * z3 * np.arctan(1000 * dq3)) /
            (1801439850948198400 * ((6177572351796469 * np.arctan((5734161139222659 * dq3 * np.arctan(1000 * dq3)) / 90071992547409920)) / 144115188075855872 + 1003 / 2500))
    )
    dz4 = (
            (32880603970571302491146779030281 * dq4 * np.arctan(1000 * dq4) ** 2) /
            81129638414606681695789005144064 -
            (13366329615528018129 * dq4 * z4 * np.arctan(1000 * dq4)) /
            (1801439850948198400 * ((6177572351796469 * np.arctan((5734161139222659 * dq4 * np.arctan(1000 * dq4)) / 90071992547409920)) / 144115188075855872 + 1003 / 2500))
    )
    dz5 = 0
    dz6 = 0
    dz = np.array([dz1, dz2, dz3, dz4, dz5, dz6]).reshape(-1, 1)

    z = z + dz * dt
    return z


# 速度求解
def Dynamics_dq(ddq, dq, dt):
    dq = dq + ddq * dt

    # limit
    dq_max = np.pi
    dq = np.clip(dq, -dq_max, dq_max)

    dq_out = np.array([dq[0], dq[1]])
    return dq_out


# 位置求解
def Dynamics_q(dq, q, dt):
    q_out = q + dq * dt
    return q_out


# 传感器
def Encoder(q):
    Delta = 2 * np.pi / 2 ** 19
    q_sample = np.zeros((2, 1))

    for i in range(2):
        Number = np.floor(q[i, 0] / Delta) + 1
        q_sample[i, 0] = Number * Delta

    return q_sample


# 控制力矩计算
def Controller(W, e13_com, e14_com, qd3, qd4, dqd3, dqd4, ddqd3, ddqd4, sq, q, dq):
    # 参数设定
    k13, k14 = 15, 15
    k23, k24 = 1, 1
    k33, k34 = 100, 100
    K1 = np.diag([k13, k14])
    K2 = np.diag([k23, k24])
    K3 = np.diag([k33, k34])

    rho_13, rho_14, rho_33, rho_34 = 0.008, 0.008, 0.5, 0.5
    rho_11, rho_12, rho_31, rho_32 = rho_13, rho_14, rho_33, rho_34

    nodes, Tau11, Tau12, eta1, b = 13, 60, 60, 0.01, 60
    c = np.array([[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30],
                  [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]])

    # 系统输入
    e1_com = np.array([e13_com, e14_com]).reshape(-1, 1)
    qd = np.array([qd3, qd4]).reshape(-1, 1)
    dqd = np.array([dqd3, dqd4]).reshape(-1, 1)
    ddqd = np.array([ddqd3, ddqd4]).reshape(-1, 1)

    # 误差信号
    # e1 = sq - qd
    e1 = sq - (qd - 0.5 * e1_com)
    de1 = dq - dqd

    # 虚拟控制器
    miu = -K1 @ e1 + dqd
    dmiu = -K1 @ de1 + ddqd

    # 辅助误差信号
    e2 = dq - miu
    e3 = K2 @ e1 + e2

    # 双环BLF
    u_BLF1 = np.array([[e1[0] / (rho_11 ** 2 - e1[0] ** 2)],
                       [e1[1] / (rho_12 ** 2 - e1[1] ** 2)]]).reshape(-1, 1)
    u_BLF2_denominator_1 = rho_31 ** 2 - e3[0] ** 2
    u_BLF2_denominator_2 = rho_32 ** 2 - e3[1] ** 2
    u_BLF2_denominator_1 = u_BLF2_denominator_1[0]
    u_BLF2_denominator_2 = u_BLF2_denominator_2[0]
    u_BLF2 = np.array([[e3[0] / u_BLF2_denominator_1],
                       [e3[1] / u_BLF2_denominator_2]]).reshape(-1, 1)
    Gain_3 = np.diag([u_BLF2_denominator_1, u_BLF2_denominator_2])

    # RBFNN
    # kernel
    xi1, xi2 = np.array([sq[0], dq[0]]), np.array([sq[1], dq[1]])  # shape[2,1] 需要检查每个shape
    h1, h2 = np.zeros((nodes, 1)), np.zeros((nodes, 1))
    for j in range(nodes):
        h1[j] = np.exp(-np.linalg.norm(xi1 - c[:, j].reshape(-1, 1)) ** 2 / (b * b))
        h2[j] = np.exp(-np.linalg.norm(xi2 - c[:, j].reshape(-1, 1)) ** 2 / (b * b))
    # predict
    W1, W2 = W[:nodes], W[nodes:2 * nodes]
    fn = np.array([W1.T @ h1, W2.T @ h2]).reshape(-1, 1)

    # 真实控制器
    q3 = q[0, 0]
    q4 = q[1, 0]

    M11 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 - (9666515293019412421 * np.sin(2 * q3 + q4)) / 147573952589676412928 - (7347855351585683 * np.cos(2 * q3)) / 18014398509481984 - (
            36945533070432904259 * np.cos(2 * q3 + 2 * q4)) / 2361183241434822606848 - (2242006792538445303 * np.sin(2 * q3 + 2 * q4)) / 1208925819614629174706176 - (
                  3932438768524053 * np.cos(2 * q3 + q4)) / 590295810358705651712 + (9666515293019412421 * np.sin(q4)) / 147573952589676412928 + 6993021903434165153791 / 9444732965739290427392
    M12 = (1566193994333460609165 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264027 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272 - (
            158738597992664517 * np.sin(q3 + np.pi / 2)) / 1152921504606846976 - 23750155206264607 / 144115188075855872
    M13 = (1566193994333460609165 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264027 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272 - (
            158738597992664517 * np.sin(q3 + np.pi / 2)) / 1152921504606846976
    M14 = (1566193994333460609165 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264027 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272
    M15 = -(3951197109736239241155 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (2242006792538445279 * np.sin(q3 + q4 + np.pi / 2)) / 1208925819614629174706176 - (
            3932438768524053 * np.cos(q3 + np.pi / 2)) / 590295810358705651712 - 2364964252341131 / 5316911983139663491615228241121378304
    M16 = (392149993849384621 * np.cos(q3 + q4 + np.pi / 2)) / 75557863725914323419136 - (260291698974264035 * np.sin(q3 + q4 + np.pi / 2)) / 151115727451828646838272
    M22 = (6599725983735443 * np.cos(q3)) / 4503599627370496 + (3932438768524053 * np.cos(q4)) / 295147905179352825856 + (9666515293019412421 * np.sin(q4)) / 73786976294838206464 + (
            8575983760191473 * np.cos(q3) * np.cos(q4)) / 590295810358705651712 + (21081034709077362555 * np.cos(q3) * np.sin(q4)) / 147573952589676412928 + (
                  21081034709077362555 * np.cos(q4) * np.sin(q3)) / 147573952589676412928 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 590295810358705651712 + 802526512685661487937 / 295147905179352825856
    M23 = (6599725983735443 * np.cos(q3)) / 9007199254740992 + (3932438768524053 * np.cos(q4)) / 295147905179352825856 + (9666515293019412421 * np.sin(q4)) / 73786976294838206464 + (
            8575983760191473 * np.cos(q3) * np.cos(q4)) / 1180591620717411303424 + (21081034709077362555 * np.cos(q3) * np.sin(q4)) / 295147905179352825856 + (
                  21081034709077362555 * np.cos(q4) * np.sin(q3)) / 295147905179352825856 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 1180591620717411303424 + 292870962545184539457 / 295147905179352825856
    M24 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (9666515293019412421 * np.sin(q4)) / 147573952589676412928 + (
            8575983760191473 * np.cos(q3) * np.cos(q4)) / 1180591620717411303424 + (21081034709077362555 * np.cos(q3) * np.sin(q4)) / 295147905179352825856 + (
                  21081034709077362555 * np.cos(q4) * np.sin(q3)) / 295147905179352825856 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 1180591620717411303424 + 36296309188291035969 / 295147905179352825856
    M25 = (2230692869490939 * np.sin(q4)) / 1152921504606846976 + (304047738388393 * np.cos(q3) * np.sin(q4)) / 144115188075855872 + (
            304047738388393 * np.cos(q4) * np.sin(q3)) / 144115188075855872 + 315508820003156273 / 2361183241434822606848
    M26 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (2962264730236869 * np.sin(q4)) / 147573952589676412928 + (8575983760191473 * np.cos(q3) * np.cos(q4)) / 1180591620717411303424 + (
            6460198293038971 * np.cos(q3) * np.sin(q4)) / 295147905179352825856 + (6460198293038971 * np.cos(q4) * np.sin(q3)) / 295147905179352825856 - (
                  8575983760191473 * np.sin(q3) * np.sin(q4)) / 1180591620717411303424 + 29720514162674078529 / 590295810358705651712
    M33 = (3932438768524053 * np.cos(q4)) / 295147905179352825856 + (9666515293019412421 * np.sin(q4)) / 73786976294838206464 + 292870962545184539457 / 295147905179352825856
    M34 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (9666515293019412421 * np.sin(q4)) / 147573952589676412928 + 36296309188291035969 / 295147905179352825856
    M35 = (2230692869490939 * np.sin(q4)) / 1152921504606846976 + 315508820003156273 / 2361183241434822606848
    M36 = (3932438768524053 * np.cos(q4)) / 590295810358705651712 + (2962264730236869 * np.sin(q4)) / 147573952589676412928 + 29720514162674078529 / 590295810358705651712
    M44 = 0.1230
    M45 = 1.3362e-04
    M46 = 0.0503
    M55 = 0.0529
    M56 = 1.2030e-06
    M66 = 0.0503

    M_66 = np.array([M11, M12, M13, M14, M15, M16,
                     M12, M22, M23, M24, M25, M26,
                     M13, M23, M33, M34, M35, M36,
                     M14, M24, M34, M44, M45, M46,
                     M15, M25, M35, M45, M55, M56,
                     M16, M26, M36, M46, M56, M66]).reshape(6, 6)

    # 核心组件
    Tau_parts = -K2 @ de1 - K3 @ e3 - fn + dmiu - Gain_3 @ u_BLF1 - np.diag([0.5, 0.5]) @ u_BLF2
    Tau = M_66 @ np.array([0, 0, Tau_parts[0][0], Tau_parts[1][0], 0, 0])

    return Tau.reshape(-1, 1)


def Dynamics_W(W, e13_com, e14_com, qd3, qd4, dqd3, dqd4, sq, dq, dt):
    k13, k14 = 15, 15
    k23, k24 = 1, 1
    k33, k34 = 100, 100
    K1 = np.diag([k13, k14])
    K2 = np.diag([k23, k24])

    rho_13, rho_14, rho_33, rho_34 = 0.008, 0.008, 0.5, 0.5
    rho_11, rho_12, rho_31, rho_32 = rho_13, rho_14, rho_33, rho_34

    nodes, Tau11, Tau12, eta1, b = 13, 60, 60, 0.01, 60
    c = np.array([[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30],
                  [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]])

    # 系统输入
    e1_com = np.array([e13_com, e14_com]).reshape(-1, 1)
    qd = np.array([qd3, qd4]).reshape(-1, 1)
    dqd = np.array([dqd3, dqd4]).reshape(-1, 1)

    #
    e1 = sq - (qd - 0.5 * e1_com)
    miu = -K1 @ e1 + dqd
    e2 = dq - miu
    e3 = K2 @ e1 + e2

    # 双环BLF
    u_BLF2_denominator_1 = rho_31 ** 2 - e3[0] ** 2
    u_BLF2_denominator_2 = rho_32 ** 2 - e3[1] ** 2
    u_BLF2_denominator_1 = u_BLF2_denominator_1[0]
    u_BLF2_denominator_2 = u_BLF2_denominator_2[0]
    u_BLF2 = np.array([[e3[0] / u_BLF2_denominator_1],
                       [e3[1] / u_BLF2_denominator_2]]).reshape(-1, 1)

    # RBFNN
    xi1, xi2 = np.array([sq[0], dq[0]]), np.array([sq[1], dq[1]])
    h1, h2 = np.zeros((nodes, 1)), np.zeros((nodes, 1))
    for j in range(nodes):
        h1[j] = np.exp(-np.linalg.norm(xi1 - c[:, j].reshape(-1, 1)) ** 2 / (b * b))
        h2[j] = np.exp(-np.linalg.norm(xi2 - c[:, j].reshape(-1, 1)) ** 2 / (b * b))
    W1, W2 = W[:nodes], W[nodes:2 * nodes]
    fn = np.array([W1.T @ h1, W2.T @ h2]).reshape(-1, 1)
    dW = np.zeros((2 * nodes, 1))
    for i in range(nodes):
        dW[i] = Tau11 * h1[i] * u_BLF2[0] - eta1 * W1[i]
        dW[nodes + i] = Tau12 * h2[i] * u_BLF2[1] - eta1 * W2[i]

    #
    NN = W + dW * dt
    return NN.reshape(-1, 1)
# 无论是matlab还是python维数一定不能错
# all variable

# q = np.array([0.1, 0.2]).reshape(-1, 1)  # 长度为 2 的数组
# sq = np.array([0.1, 0.2]).reshape(-1, 1)  # 长度为 2 的数组
# dq = np.array([0.4, 0.6]).reshape(-1, 1)  # 长度为 2 的数组
# ddq = np.array([0.44, 0.64]).reshape(-1, 1)  # 长度为 2 的数组
# torque = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).reshape(-1, 1)
# z = np.array([0.1, 0.2, 0.7, 0.8, 0.5, 0.6]).reshape(-1, 1)
# dt = 0.123
# W = np.ones((26,1))  # 长度为 26，全为 1 的数组
# e13_com, e14_com = 0.5, 0.7  # 设定 e13_com 和 e14_com 的值
# qd3, qd4, dqd3, dqd4, ddqd3, ddqd4 = 0.3, 0.4, 0.2, 0.1, 0.6, 0.8  # 设定其他值

# res_ddq = Dynamics_ddq(torque,q,dq,z)

#
# res_dq = Dynamics_z(dq,z,dt)
# for i in range(6):
#     print('{};'.format(res_dq[i][0]))

# res_dq2 = Dynamics_dq(ddq,dq,dt)
# for i in range(2):
#     print('{};'.format(res_dq2[i][0]))

# res_q = Dynamics_q(dq,q,dt)
# for i in range(2):
#     print('{};'.format(res_q[i][0]))

# res_q_s = Encoder(q)
# for i in range(2):
#     print('{};'.format(res_q_s[i][0]))


# res_tau = Controller(W,e13_com,e14_com,qd3,qd4,dqd3,dqd4,ddqd3,ddqd4,sq,q,dq)
# for i in range(6):
#     print('{};'.format(res_tau[i][0]))

# res_NN = Dynamics_W(W,e13_com,e14_com,qd3,qd4,dqd3,dqd4,sq,dq, dt)
# for i in range(26):
#     print('{};'.format(res_NN[i][0]))
# a = 0
