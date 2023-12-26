import torch.optim as optim
import torch.nn.functional as F
import torch
from test_model.dnn import lstm_n
from test_model.block import ReplayBuffer
import torch.optim.lr_scheduler as lr_scheduler


class dynamic_lr:
    def __init__(self, optimizer, mode='min', factor=0.1, improve_factor=1.1, patience=5, max_lr=1e-2,  min_lr=1e-4, verbose=False, improve_patience=5, lr=0.001):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.improve_factor = improve_factor
        self.patience = patience
        self.improve_patience = improve_patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.lr = lr

        self.best_loss = float('inf') if mode == 'min' else -float('inf')
        self.improve_counter = 0
        self.counter = 0


    def step(self, current_loss):
        if self.mode == 'min':
            improved = current_loss < self.best_loss
        else:
            improved = current_loss > self.best_loss

        if improved:
            self.best_loss = current_loss
            self.counter = 0
            self.improve_counter += 1
            if self.improve_counter >= self.improve_patience:
                self.improve_counter = 0
                self.lr = self.lr * self.improve_factor
                if self.lr < self.max_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
                    if self.verbose:
                        print(f'imporve learning rate updated to {self.optimizer.param_groups[0]["lr"]}')
        else:
            self.counter += 1
            self.improve_counter = 0
            if self.counter >= self.patience:
                self.counter = 0
                self.lr = self.lr * self.factor
                if self.lr > self.min_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
                    if self.verbose:
                        print(f'lower learning rate updated to {self.optimizer.param_groups[0]["lr"]}')
class dynamic_lr2:
    '''

    初始时候先固定一个，之后再用其他的
    '''
    def __init__(self, optimizer, mode='min', factor=0.1, improve_factor=1.1, patience=5, max_lr=1e-2,  min_lr=1e-4, verbose=False, improve_patience=5, lr=0.001):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.improve_factor = improve_factor
        self.patience = patience
        self.improve_patience = improve_patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.lr = lr

        self.best_loss = float('inf') if mode == 'min' else -float('inf')
        self.improve_counter = 0
        self.counter = 0


    def step(self, current_loss):
        if self.mode == 'min':
            improved = current_loss < self.best_loss
        else:
            improved = current_loss > self.best_loss

        if current_loss > 100:
            self.lr = 0.001
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            if self.verbose:
                print(f'set learning rate updated to {self.optimizer.param_groups[0]["lr"]}')
        elif current_loss > 10:
            self.lr = 0.0001
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            if self.verbose:
                print(f'set learning rate updated to {self.optimizer.param_groups[0]["lr"]}')
        if improved:
            self.counter = 0
            self.best_loss = current_loss
        else:
            self.counter += 1
            self.improve_counter = 0
            if self.counter >= self.patience:
                self.counter = 0
                self.lr = self.lr * self.factor
                if self.lr > self.min_lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
                    if self.verbose:
                        print(f'lower learning rate updated to {self.optimizer.param_groups[0]["lr"]}')

class exp_model:
    ''' model for online training'''

    def __init__(self, args, Model):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.model = Model(args).float().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.999, patience=200, verbose=True)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.8)
        # self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500, eta_min=args.lr*0.01)
        # self.scheduler = dynamic_lr2(self.optimizer, mode='min', factor=0.99, improve_factor=1.1, patience=50, max_lr=1e-2,  min_lr=1e-7, verbose=True, improve_patience=5, lr=args.lr)
        self.args = args

    def train_one_epoch(self):
        self.model.train()
        if self.args.sample_type == 'log':
            inputs, labels = self.replay_buffer.log_sample(self.args.batch_size)
        elif self.args.sample_type == 'linear':
            #  # 使用linear sample时候buffer size == batch_size
            inputs, labels = self.replay_buffer.linear_sample(self.args.batch_size)
        elif self.args.sample_type == 'random':
            inputs, labels = self.replay_buffer.sample(self.args.batch_size)
        inputs = torch.from_numpy(inputs).to(torch.float32).to(self.device)  # .requires_grad_(True)
        labels = torch.from_numpy(labels).to(torch.float32).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, labels)
        # print(loss.item())
        loss.backward()
        self.optimizer.step()
        lr_prev = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(loss.item())
        lr = self.optimizer.param_groups[0]['lr']
        print(loss.item())
        # if lr != lr_prev:
        #     print('Updating learning rate to {}'.format(lr))

    def pred(self, current_inputs):
        self.model.eval()
        current_inputs = torch.from_numpy(current_inputs).to(torch.float32).to(self.device)
        with torch.no_grad():
            pred = self.model(current_inputs.unsqueeze(0)).cpu().detach().numpy().transpose()
        return pred

    def update_buffer(self, total_info, total_label):  # 将数据加入buffer
        self.replay_buffer.add(total_info, total_label)

    def get_buffer_size(self):
        return self.replay_buffer.size()
