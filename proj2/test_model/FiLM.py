import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import special as ss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 对应公式3，感觉有点位置编码的意思
'''
A:[windows_size,windows_size]
B:[windows_size,1]
'''
def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)  # 复制Q，生成二维向量，j和i分别是网格的横纵坐标
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R  # where 返回的是index
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT, self).__init__()
        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # 将连续时间系统的传递函数转换为离散时间系统的差分方程
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device))
        self.register_buffer('B', torch.Tensor(B).to(device))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))

    def forward(self, inputs):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection

        ##
        legt变化只与pred_len相关，但是他是从seq_len中截取的后pred_len个数值，然后相当于每个数值增广了window_size那么多的维数
        input:[batch_size,enc_in,x*pred_len]
        output:[x*pred_len,batch_size,enc_in,self.N]
        A:[windows_size,windows_size]
        B:[windows_size]
        """
        # 两个tuple的拼接
        # 下面代表蓝色公式
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)  # [batch_size,enc_in,1]
            new = f @ self.B.unsqueeze(0)  # @矩阵乘法 [batch_size,enc_in,N]
            c = F.linear(c, self.A) + new  # [batch_size,enc_in,N]
            cs.append(c)
        # 将list中所有元素展开，dim=len(list)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        # 这里的可学习参数是为了保证计算前后的shape一致，这样效果剩余使用几个不同的linear进行变换
        self.weights_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))   # [window_size,window_size,min(32,seq_len,pred_len]
        self.weights_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        # 创建一个复数向量前面是实部，后面是虚部
        # 这里就是一个简单的复数乘法，不过是结合了神经网络之后的
        return torch.complex(torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
                             torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real))

    def forward(self, x):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)  # x_ft是一个复数矩阵，沿着最后一个维度，下面选择前mode个点是不是就是low rank approximation，因为当pred_len非常长的时候，a将会是一个相对来说比较小的数值
        # 这里是因为需要进行傅里叶变换，所以x.size(-1)//2+1
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]  # [batch_size,enc_in,N,:mode]代表的是0~mode
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)  # 可以消去任意一项两者中共同出现的字母，并进行排序 # i=o 所以这里并没有改变形状
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2205.08897
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        # self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.pred_len = configs.seq_len if configs.pred_len == 0 else configs.pred_len


        self.output_attention = configs.output_attention
        self.layers = configs.e_layers
        self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        # b, s, f means b, f
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

        self.multiscale = [1, 2, 4]
        self.window_size = [256]
        configs.ratio = 0.5
        self.legts = nn.ModuleList([HiPPO_LegT(N=n, dt=1. / self.pred_len / i) for n in self.window_size for i in self.multiscale])
        self.spec_conv_1 = nn.ModuleList([SpectralConv1d(in_channels=n, out_channels=n, seq_len=min(self.pred_len, self.seq_len), ratio=configs.ratio)
                                          for n in self.window_size for _ in range(len(self.multiscale))])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)



    def forward(self, x_enc):
        # Normalization from Non-stationary Transformer
        # means = x_enc.mean(1, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        # x_enc /= stdev

        # x_enc = x_enc * self.affine_weight + self.affine_bias
        x_decs = []
        jump_dist = 0  # 没被修改过
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            # 周期性的依照multiscale里面的设置，输出一个为pred_len倍数的数值，lookback windows 的长度
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.pred_len
            x_in = x_enc[:, -x_in_len:]
            legt = self.legts[i]
            # [batch_size, enc_in, self.N, x*pred_len]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            # [batch_size, enc_in, self.N, x*pred_len]
            out1 = self.spec_conv_1[i](x_in_c)
            # 选择和pred_len位置的N，或者选最后一个
            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:, :, self.pred_len - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            # 为了从映射中进行重建 #[batch_size,enc_in,seq_len]
            x_dec = x_dec_c @ legt.eval_matrix[-self.pred_len:, :].T
            x_decs.append(x_dec)
        x_dec = torch.stack(x_decs, dim=-1)  # 堆叠起来，方便下一步直接mlp融合多层的信息
        x_dec = self.mlp(x_dec).squeeze(-1).permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer # 这里有些许不同
        # x_dec = x_dec - self.affine_bias
        # x_dec = x_dec / (self.affine_weight + 1e-10)
        # x_dec = x_dec * stdev
        # x_dec = x_dec + means
        return x_dec
