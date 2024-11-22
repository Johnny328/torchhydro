from typing import Tuple
import torch
import torch.nn as nn

class BALSTM(nn.Module):
    def __init__(self, input_size_sta, input_size_dyn, input_size_glo, hidden_size, output_size, num_layers, dropout=0.5,prec_window=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 使用SingleBALSTM作为子模块
        self.lstm = SingleBALSTM(input_size_sta, input_size_dyn, input_size_glo, hidden_size,
                                 batch_first=True, initial_forget_bias=0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
    # def forward(self, data):
    #     x_s, x_d, x_g = data
    def forward(self, x_s, x_d, x_g):
        if x_s.dim() == 3:
            # x_s = x_s.view(32,-1)
            x_s=x_s.squeeze(1)
        # 前向传播LSTM
        # out: tensor的形状为(batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x_s, x_d, x_g)
        # # 解码最后一个时间步的隐藏状态
        # out = self.fc(self.dropout(out[:, -1, :]))
        out = self.fc(self.dropout(out))
        # # 使用特定激活函数
        out = self.act(out)
        return out


class SingleBALSTM(nn.Module):
    """实现基于 Basin-Aware-LSTM (BA-LSTM) 的模型"""

    def __init__(self,
                 input_size_sta: int,
                 input_size_dyn: int,
                 input_size_glo: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super().__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_sta = input_size_sta
        self.input_size_glo = input_size_glo
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # 创建可学习参数的张量
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size_dyn, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(input_size_sta, 2 * hidden_size))
        self.weight_gh = nn.Parameter(torch.FloatTensor(input_size_glo, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(2 * hidden_size))
        self.bias_g = nn.Parameter(torch.FloatTensor(hidden_size))

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 LSTM 的所有可学习参数"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_gh.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias!= 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_s: torch.Tensor, x_d: torch.Tensor, x_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_d : torch.Tensor
            包含动态特征序列的批量张量。形状必须与 batch_first 指定的格式匹配。
        x_s : torch.Tensor
            包含静态特征的批量张量。
        x_g : torch.Tensor
            包含全局特征的批量张量。

        Returns
        -------
        h_n : torch.Tensor
            批量中每个样本每个时间步的隐藏状态。
        c_n : torch.Tensor
            批量中每个样本每个时间步的记忆状态。
        """
        if self.batch_first:
            # 交换维度顺序为 (seq_len, batch_size, feature_size)
            x_d = x_d.transpose(0, 1)
            x_g = x_g.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        h_n, c_n = [], []

        bias_batch = self.bias.unsqueeze(0).expand(batch_size, *self.bias.size())
        bias_g_batch = self.bias_g.unsqueeze(0).expand(batch_size, *self.bias_g.size())
        
        bias_s_batch = self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size())
        #0920修改
        # bias_s_batch = self.bias_s.unsqueeze(0).unsqueeze(0).expand(seq_len, batch_size, -1)
        i = torch.addmm(bias_s_batch, x_s, self.weight_sh)
        i1, i2 = i.chunk(2, 1)
        i1 = torch.sigmoid(i1)
        i2 = torch.sigmoid(i2)

        for t in range(seq_len):
            h_0, c_0 = h_x

            g = torch.addmm(bias_g_batch, x_g[t], self.weight_gh)
            gates = torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x_d[t], self.weight_ih)
            f, d, o_h = gates.chunk(3, 1)
            o = o_h + torch.mm(x_g[t], self.weight_gh)
            c_1 = torch.sigmoid(f) * c_0 + i1 * torch.tanh(d) + i2 * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            # 根据 batch_first 参数调整输出维度顺序
            return h_n.transpose(1, 0), c_n.transpose(1, 0)
        else:
            return h_n, c_n
