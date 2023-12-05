import numpy as np
import torch
import matplotlib.pyplot as plt
import math
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, scheduler=None, val_loss=0):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        lr_adjust = {epoch: args.learning_rate / epoch}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_prev = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        if lr != lr_prev:
            print('Updating learning rate to {}'.format(lr))
        return
    elif args.lradj == 'type4':
        lr_prev = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        if lr != lr_prev:
            print('Updating learning rate to {}'.format(lr))
        return
    if args.lradj == 'type5':
        print('type5 does not update learning rate')
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization

    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def extract_seq(data, pred_len):
    # 这一行的意思是间隔是seq_len,每次都获取当前最大的预测长度填进去，而不是每次只是简单的使用紧接着那个数值
    # 注意区分这个和data[:pred_len, :, :]
    # data = [test_num/pred_len,:,:]
    data = data[::pred_len, :, :]
    # print(data.shape)
    data = data.reshape(-1, data.shape[-1])  # 保持维数不变展平预测的部分
    return data


def visual_all(res_true, res_pred, name, c_out):
    fig, axs = plt.subplots(2, math.ceil(c_out/2), figsize=(12, 6))
    idx = 0
    # 这里有个bug就是，如果列为1，那么就会导致没有len(axs[0])
    if math.ceil(c_out/2) > 1:
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
