import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        # 两层norm间是feed forward # transpose(-1,1) 的意思是交换最后一个和第一个维度，为了将seq_len换到最后，卷积是在seq_len上面去做的
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class feature_decompose(nn.Module):
    def __init__(self, in_channels, seq_len, pred_len, configs, individual=False):
        """
        seq_len是输入的长度
        pred_len是输出长度，这里我们只作为编码器使用所以是恒等输入，pred=seq_len
        decompose into 2
        dlinear只使用了一层线性神经网络
        """
        super(feature_decompose, self).__init__()
        self.decompsition = series_decomp(configs.moving_avg)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = in_channels
        # from DLinear
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    # input [batch_size, d_model, seq_len]
    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)  # [batch_size,seq_len, enc_in]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # [batch_size, enc_in, seq_len]
        if self.individual:
            # [batch_size, enc_in, pred_len]
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)


class feature_enhanced(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, ratio=0.5):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(feature_enhanced, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        # 这里的可学习参数是为了保证计算前后的shape一致，这样效果剩余使用几个不同的linear进行变换
        self.weights_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))  # [window_size,window_size,min(32,seq_len,pred_len]
        self.weights_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order, x, weights_real, weights_imag):
        # 创建一个复数向量前面是实部，后面是虚部
        # 这里就是一个简单的复数乘法，不过是结合了神经网络之后的
        return torch.complex(torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
                             torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real))

    # input [batch_size, d_model, seq_len]
    def forward(self, x):
        batch_size, channel, seq_len = x.shape
        x_ft = torch.fft.rfft(x)  # x_ft是一个复数矩阵，沿着最后一个维度，下面选择前mode个点是不是就是low rank approximation，因为当pred_len非常长的时候，a将会是一个相对来说比较小的数值
        # 这里是因为需要进行傅里叶变换，所以x.size(-1)//2+1
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :self.modes]  # [batch_size,enc_in,N,:mode]代表的是0~mode
        out_ft[:, :, :self.modes] = self.compl_mul1d("bix,iox->box", a, self.weights_real, self.weights_imag)  # 可以消去任意一项两者中共同出现的字母，并进行排序 # i=o 所以这里并没有改变形状
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class EncoderLayer_f(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", args=None, is_inversed=True):
        super(EncoderLayer_f, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.is_inversed = is_inversed
        if is_inversed:
            # 注意如果是有x_mark的c_out要加一定数值，ETT的是加5
            self.fe = feature_enhanced(in_channels=args.c_out, out_channels=args.c_out, seq_len=d_model)
        else:
            self.fe = feature_enhanced(in_channels=d_model, out_channels=d_model, seq_len=args.seq_len)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        # 两层norm间是feed forward
        # conv1d是对第二个维度，也就是通道方面做得变换
        # 对于标准transformer[batch_size,seq_len,d_model]->[batch_size,d_ff,seq_len]->[batch_size,seq_len,d_model]
        # 对于inversed来说 [batch_size,c_out+x_mark_cout,d_model],d_model代表的是seq_len方向->[batch_size,d_ff,c_out++]->[batch_size,c_out++,d_model]
        if self.is_inversed:
            # 使用fe时候需要第三个维度是seq_len
            y = self.fe(y)
        else:
            y = self.fe(y.transpose(-1, 1)).transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class EncoderLayer_d(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", args=None, is_inversed=True):
        super(EncoderLayer_d, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.is_inversed = is_inversed
        if is_inversed:
            # 注意如果是有x_mark的c_out要加一定数值，ETT的是加5
            # 传入args只是为了move_avg
            self.fd = feature_decompose(in_channels=args.c_out, seq_len=d_model, pred_len=d_model, configs=args)
        else:
            self.fd = feature_decompose(in_channels=d_model, seq_len=args.seq_len, pred_len=args.seq_len, configs=args)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        # 两层norm间是feed forward
        # conv1d是对第二个维度，也就是通道方面做得变换
        # 对于标准transformer[batch_size,seq_len,d_model]->[batch_size,d_ff,seq_len]->[batch_size,seq_len,d_model]
        # 对于inversed来说 [batch_size,c_out+x_mark_cout,d_model],d_model代表的是seq_len方向->[batch_size,d_ff,c_out++]->[batch_size,c_out++,d_model]
        if self.is_inversed:
            # 使用fe时候需要第三个维度是seq_len
            y = self.fd(y.transpose(-1, 1)).transpose(-1, 1)
        else:
            y = self.fd(y)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
