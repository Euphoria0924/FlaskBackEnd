import torch
import torch.nn as nn
from fftKAN import *
from effKAN import *


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的数量

        # LSTM网络层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，用于将LSTM的输出转换为最终的输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        # 前向传播LSTM，返回输出和最新的隐藏状态与细胞状态
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 将LSTM的最后一个时间步的输出通过全连接层
        out = self.fc(out[:, -1, :])
        return out
class Attention(nn.Module):
    """Attention Mechanism"""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_length, hidden_size]
        attn_weights = torch.tanh(self.attn(lstm_output))  # [batch_size, seq_length, 1]
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        # Apply attention weights
        new_hidden_state = torch.sum(lstm_output * soft_attn_weights, dim=1)
        return new_hidden_state


class LSTMAttention_ekan(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout):
        super(LSTMAttention_ekan, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.e_kan = KAN([hidden_dim, output_dim])

    def forward(self, x):
        # 数据先输入到LSTM 再经过一个Attention，然后经过一个线性层输出，可调整模型结构顺序。例如在attention后面再加一层lstm
        # x: [batch_size, seq_length, input_dim]
        lstm_out, (hn, cn) = self.lstm(x)
        # lstm_out: [batch_size, seq_length, hidden_dim]
        attn_out = self.attention(lstm_out)
        # attn_out: [batch_size, hidden_dim]
        out = self.e_kan(attn_out)

        return out
















