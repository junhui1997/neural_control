import torch.optim as optim
import torch.nn.functional as F
import torch
from test_model.dnn import lstm_n
from test_model.block import ReplayBuffer

class exp_model:
    ''' 经验回放池 '''

    def __init__(self, args, Model ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.model = Model(args).float().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr)
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