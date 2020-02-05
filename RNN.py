import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, batch_size, hidden_size, time_window, device):
        super(RNN, self).__init__()
        self.b_n = batch_size
        self.s_n = hidden_size
        self.T_n = time_window
        self.device = device
        self.cell = nn.RNN(input_size=time_window, hidden_size=hidden_size, batch_first=True)
    def forward(self, x, hidden):
        x = x.view(self.b_n, self.s_n, self.T_n)
        out, hidden = self.cell(x, hidden)
        out = out.view(-1, 2)
        return out, hidden
    def init_hidden(self):
        # initialize hidden and cell states
        return torch.zeros(1, self.b_n, self.s_n).to(self.device)

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, drop_prob=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = F.relu(out[:, -1])
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden