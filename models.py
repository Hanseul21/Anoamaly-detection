import torch.nn as nn
import torch
import pywt
import numpy as np
from utils import View

class GRU_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, n_layers=1, drop_prob=0.2, device='cuda'):
        super(GRU_AE, self).__init__()
        self.hidden_dim = hidden_dim
        self.k = k
        self.n_layers = n_layers
        self.device = device
        self.in_gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.out_gru = nn.GRU(k, k, n_layers, batch_first=True, dropout=drop_prob)

        self.linear01 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear02 = nn.Linear(hidden_dim, k, bias=True)

        self.linear03 = nn.Linear(k, k, bias=True)
        self.linear04 = nn.Linear(k, hidden_dim, bias=True)
        self.linear05 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.linear11 = nn.Linear(k, k, bias=True)
        self.linear12 = nn.Linear(k, k, bias=True)

        self.linear21 = nn.Linear(hidden_dim, input_dim, bias=True)
        self.linear22 = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x, h1, h12, h2):
        q_size = x.size(1)
        z, z_mu, z_logvar = self.encoding(x, h1)
        x_mu, x_logvar = self.decoding(z, h2, q_size)
        return x_mu.squeeze(), x_logvar.squeeze(), z_mu, z_logvar, z

    def encoding(self, x, h1):
        z_, h = self.in_gru(x, h1)
        z_ = torch.tanh(z_[:,-1])   # batch, hidden
        z_ = z_.contiguous().view(-1, self.hidden_dim)
        z_ = torch.tanh(self.linear01(z_))
        z_ = torch.tanh(self.linear02(z_))
        z_mu = self.linear11(z_)
        z_logvar = self.linear12(z_)
        z_std = z_logvar.mul(0.5).exp()

        z = self.reparameterizing(z_mu, z_std)
        return z_, z_mu, z_logvar

    def decoding(self, z, h2, q_size):
        # x_ = torch.relu(z)

        x_ = z.view(-1, 1, self.k)

        # x_ = x_.view(-1, 30, self.hidden_dim)
        recon_x = []
        for i in range(q_size):
            x_, h2 = self.out_gru(x_, h2)
            x_ = torch.tan(self.linear03(x_))
            if i == 0:
                recon_x = x_
            else:
                recon_x = torch.cat((recon_x, x_), dim=1)
        recon_x = torch.tanh(self.linear04(recon_x))
        # recon_x = torch.relu(self.linear05(recon_x))
        x_mu = self.linear21(recon_x)   # Batch, Time, k
        x_logvar = self.linear22(recon_x)

        return x_mu, x_logvar

    def reparameterizing(self, mu, std):
        z = mu + std.mul(torch.randn_like(std))
        return z

    def init_hidden(self, batch_size, hidden_dim):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, hidden_dim).zero_().to(self.device)
        return hidden

class VAENet(nn.Module):
    def __init__(self, input_n, k, output_size, hidden_size, device):
        super(VAENet, self).__init__()
        self.device = device
        self.output_size = output_size
        # encoding
        self.linear_1 = nn.Linear(input_n, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, hidden_size)
        self.linear_5 = nn.Linear(hidden_size, hidden_size)
        # mu & std
        self.linear_31 = nn.Linear(hidden_size, k)
        self.linear_32 = nn.Linear(hidden_size, k)
        # decoding
        self.linear_6 = nn.Linear(k, hidden_size)
        self.linear_7 = nn.Linear(hidden_size, hidden_size)
        self.linear_8 = nn.Linear(hidden_size, hidden_size)
        self.linear_9 = nn.Linear(hidden_size, hidden_size)
        self.linear_10 = nn.Linear(hidden_size, hidden_size)

        self.linear_61 = nn.Linear(hidden_size, output_size)
        self.linear_62 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        z_mu, z_logvar = self.encoding(x)
        z = self.reparametrizing(z_mu, z_logvar)
        x_mu, x_logvar = self.decoding(z)
        return x_mu, x_logvar, z_mu, z_logvar, z

    def encoding(self, x):
        x_ = torch.tanh(self.linear_1(x))
        x_ = torch.tanh(self.linear_2(x_))
        x_ = torch.tanh(self.linear_3(x_))
        x_ = torch.tanh(self.linear_4(x_))
        x_ = torch.tanh(self.linear_5(x_))
        z_mu = self.linear_31(x_)
        z_logvar = self.linear_32(x_)
        return z_mu, z_logvar

    def reparametrizing(self, mu, log_var):
        std = log_var.mul(0.5).exp()
        eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).to(self.device)
        z = std.mul(eps).add_(mu)
        return z

    def decoding(self, z):
        z_ = torch.tanh(self.linear_6(z))
        z_ = torch.tanh(self.linear_7(z_))
        z_ = torch.tanh(self.linear_8(z_))
        z_ = torch.tanh(self.linear_9(z_))
        z_ = torch.tanh(self.linear_10(z_))
        x_mu = self.linear_61(z_)
        x_logvar = (self.linear_62(z_))
        return x_mu, x_logvar

class GRU_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, n_layers=1, drop_prob=0.2, device='cuda'):
        super(GRU_VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.in_gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.in_gru2 = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.out_gru = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.out_gru2 = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.linear01 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear02 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear03 = nn.Linear(k, hidden_dim, bias=True)
        self.linear04 = nn.Linear(hidden_dim, hidden_dim*30, bias=True)

        self.linear11 = nn.Linear(hidden_dim, k, bias=True)
        self.linear12 = nn.Linear(hidden_dim, k, bias=True)

        self.linear21 = nn.Linear(hidden_dim, input_dim, bias=True)
        self.linear22 = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, x, h1, h12, h2):
        z, z_mu, z_logvar = self.encoding(x, h1)
        x_mu, x_logvar = self.decoding(z, h2, h12)
        # return x_mu.squeeze(), x_logvar.squeeze(), z_mu, z_logvar, z
        return x_mu.squeeze(), x_logvar.squeeze(), z_mu, z_logvar, z

    def encoding(self, x, h1):
        z_, h = self.in_gru(x, h1)
        z_ = torch.relu(z_[:,-1])   # batch, hidden
        z_ = z_.contiguous().view(-1, self.hidden_dim)
        z_ = torch.relu(self.linear01(z_))
        # z_ = torch.relu(self.linear02(z_))
        z_mu = self.linear11(z_)
        # z_, h = self.in_gru2(z_, h12)
        z_logvar = self.linear12(z_)
        # # z_mu = torch.sigmoid(z_)
        # z_mu = torch.sigmoid(self.linear11(z_))
        # z_logvar = torch.sigmoid(self.linear12(z_))
        # # z_logvar = torch.sigmoid(z_)
        z_std = z_logvar.mul(0.5).exp()

        z = self.reparameterizing(z_mu, z_std)
        return z, z_mu, z_logvar

    def decoding(self, z, h2, h12):
        x_ = torch.relu(z)
        x_ = torch.relu(self.linear03(x_))
        x_ = torch.relu(self.linear04(x_))

        x_ = x_.view(-1, 30, self.hidden_dim)
        # z = self.linear21(x_)
        # z_ = self.linear22(x_)

        x_, h = self.out_gru(x_, h2)
        x_mu = self.linear21(x_)
        # x_logvar, h = self.out_gru2(x_, h12)
        x_logvar = self.linear22(x_)

        return x_mu, x_logvar

    def reparameterizing(self, mu, std):
        z = mu + std.mul(torch.randn_like(std))
        return z

    def init_hidden(self, batch_size, hidden_dim):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, hidden_dim).zero_().to(self.device)
        return hidden

class ConvAE(nn.Module):
    def __init__(self, k, in_channels):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  #B, 513, 14
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(num_features=64),    #B, 64, 7
            nn.ReLU(True),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(num_features=128),   #B, 128, 4
            nn.ReLU(True),
            nn.Conv1d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm1d(num_features=256),   #B, 256, 2
            nn.ReLU(True),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            View((-1, 256*2)),
            nn.Linear(256*2, k),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(k, 256*2),
            View((-1, 256, 2)),                        # B, 256, 2
            nn.ConvTranspose1d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               stride=2,
                               padding=0,
                               bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=128,  # B, 128, 5
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=0,
                               bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64,          # B, 64, 11
                               out_channels=in_channels,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
        )                                               # B, 514, 14

class Wavelet(nn.Module):
    def __init__(self, in_channels, q_size, k):
        super(Wavelet, self).__init__()
        self.q_size = q_size
        self.wavelet_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 513, 14
                      out_channels=1,
                      kernel_size=31,
                      stride=1,
                      padding=15,
                      bias=False),  # B, 64, 7
            # nn.BatchNorm1d(num_features=1),
            nn.Sigmoid())
        self.wavelet_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 513, 14
                      out_channels=1,
                      kernel_size=11,
                      stride=1,
                      padding=5,
                      bias=False),  # B, 64, 7
            # nn.BatchNorm1d(num_features=1),
            nn.Sigmoid())
        self.wavelet_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 513, 14
                      out_channels=1,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=False),  # B, 64, 7
            # nn.BatchNorm1d(num_features=1),
            nn.Sigmoid())
        self.fc1 = nn.Linear(q_size, k, bias=True)
        self.fc11 = nn.Linear(k, q_size, bias=True)
        self.fc112 = nn.Linear(q_size, q_size, bias=True)
        self.fc12 = nn.Linear(q_size, q_size, bias=True)
        self.fc2 = nn.Linear(q_size, k, bias=True)
        self.fc21 = nn.Linear(k, q_size, bias=True)
        self.fc212 = nn.Linear(q_size, q_size, bias=True)
        self.fc22 = nn.Linear(q_size, q_size, bias=True)
        # self.fc3 = nn.Linear(3, 1, bias=True)

    def forward(self, x):
        coef_1 = self.wavelet_1(x)
        coef_1 = coef_1.view(-1, self.q_size)
        coef_1_z = torch.tanh(self.fc1(coef_1))
        coef_1 = torch.tanh(self.fc11(coef_1_z))
        coef_1 = torch.tanh(self.fc112(coef_1))
        coef_1 = self.fc12(coef_1)

        # coef_2 = self.wavelet_2(x)
        coef_3 = self.wavelet_3(x)
        coef_3 = coef_3.view(-1, self.q_size)
        coef_3_z = torch.tanh(self.fc2(coef_3))
        coef_3 = torch.tanh(self.fc21(coef_3_z))
        coef_3 = torch.tanh(self.fc212(coef_3))
        coef_3 = self.fc22(coef_3)
        # # coeff = (coef_3 + coef_2 + coef_1)/3
        # # coeff = coeff.view(-1,self.q_size, 1)
        #
        # # z = torch.cat([coef_1, coef_2, coef_3], dim=1)
        #
        #
        #
        # z = z.view(-1, self.q_size, 3)
        # signal = torch.relu(self.fc(z))
        # signal = torch.tanh(self.fc2(signal))
        # output = self.fc3(signal)
        # # x = x.view(-1, self.q_size, 1)
        return coef_1, coef_3, coef_1_z, coef_3_z

class Wavelet_VAE(nn.Module):
    def __init__(self, in_channels, q_size, k):
        super(Wavelet_VAE, self).__init__()
        self.q_size = q_size
        self.wavelet_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  #B, 513, 14
                      out_channels=1,
                      kernel_size=31,
                      stride=1,
                      padding=15,
                      bias=False),    #B, 64, 7
            nn.Tanh())
        self.wavelet_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 513, 14
                      out_channels=1,
                      kernel_size=11,
                      stride=1,
                      padding=5,
                      bias=False),  # B, 64, 7
            nn.Tanh())
        self.wavelet_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 513, 14
                      out_channels=1,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=False),  # B, 64, 7
            nn.Tanh())
        self.wavelet_4 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  #B, 513, 14
                      out_channels=1,
                      kernel_size=21,
                      stride=1,
                      padding=10,
                      bias=True),    #B, 64, 7
            nn.Tanh())
        self.wavelet_5 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 513, 14
                      out_channels=1,
                      kernel_size=15,
                      stride=1,
                      padding=7,
                      bias=True),  # B, 64, 7
            nn.Tanh())
        self.wavelet_6 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 513, 14
                      out_channels=1,
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=True),  # B, 64, 7
            nn.Tanh())
        self.fc1 = nn.Linear(self.q_size * 3,self.q_size * 3)
        self.fc12 = nn.Linear(self.q_size * 3, self.q_size * 3)
        self.fc21 = nn.Linear(self.q_size * 3, k)
        self.fc22 = nn.Linear(self.q_size * 3, k)
        self.fc3 = nn.Linear(k, self.q_size)
        self.fc32 = nn.Linear(self.q_size, self.q_size)
        self.fc41 = nn.Linear(self.q_size, self.q_size)
        self.fc42 = nn.Linear(self.q_size, self.q_size)
    def forward(self, x):
        coef_1 = self.wavelet_1(x)
        coef_2 = self.wavelet_2(x)
        coef_3 = self.wavelet_3(x)
        # coef_4 = self.wavelet_4(x)
        # coef_5 = self.wavelet_5(x)
        # coef_6 = self.wavelet_6(x)

        # coeff = (coef_3 + coef_2 + coef_1)/3
        # coeff = coeff.view(-1,self.q_size, 1)
        signal = torch.cat([coef_1, coef_2, coef_3],dim=1)
        signal = signal.view(-1, self.q_size * 3)
        z_ = torch.relu(self.fc1(signal))
        z_ = torch.tanh(self.fc12(z_))
        z_mu = self.fc21(z_)
        z_logvar = self.fc22(z_)

        z_std = z_logvar.mul(0.5).exp()
        z = self.reparameterizing(z_mu, z_std)
        z_ = torch.relu(self.fc3(z))
        z_ = torch.tanh(self.fc32(z_))
        x_mu =self.fc41(z_)
        x_logvar = self.fc42(z_)

        return x_mu, x_logvar, z_mu, z_logvar, z

    def reparameterizing(self, mu, std):
        z = mu + std.mul(torch.randn_like(std))
        return z

class Conv_VAE(nn.Module):
    def __init__(self, in_channels, q_size, k):
        super(Conv_VAE, self).__init__()
        self.q_size = q_size
        self.h = self.q_size # 15 -1 + 15 -1
        self.k = k
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,  # B, 120, 1
                      out_channels=k,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=k),
            )
        self.conv_2 = nn.Sequential(            # B, 106, k
            nn.Conv1d(in_channels=k,
                      out_channels=k,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=k),
            )
        self.conv_3 = nn.Sequential(  # B, 106, k
            nn.Conv1d(in_channels=k,
                      out_channels=k,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=k),
        )
        self.conv_4 = nn.Sequential(  # B, 106, k
            nn.Conv1d(in_channels=k,
                      out_channels=k,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=k),
        )
        self.conv_5 = nn.Sequential(  # B, 106, k
            nn.Conv1d(in_channels=k,
                      out_channels=k,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=k),
        )
        self.deconv_1 = nn.Sequential(
            View((-1, k, self.h)),                  # B, 92, k
            nn.ConvTranspose1d(in_channels=k,
                               out_channels=k,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=True),
            nn.BatchNorm1d(k),
            nn.Sigmoid())
        self.deconv_2_1 = nn.Sequential(            # B, 62, k
            nn.ConvTranspose1d(in_channels=k,
                               out_channels=in_channels,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=True),)
        self.deconv_2_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=k,
                               out_channels=in_channels,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=True),)
        self.fc_z = nn.Linear(5*k*q_size, 5*k, bias=True)
        self.fc_z2 = nn.Linear(5*k, 5*k, bias=True)
        self.fc_z_mu = nn.Linear(5*k, 3, bias=True)     # B, 62, k  -> B, 2, k
        self.fc_z_logvar = nn.Linear(5*k, 3, bias=True)  # B, 62, k -> B, 2, k
        self.fc_dec0 = nn.Linear(3, q_size, bias=True)
        self.fc_dec1 = nn.Linear(q_size, q_size, bias=True)
        self.fc_dec = nn.Linear(q_size, q_size, bias=True)
        self.fc_dec2 = nn.Linear(q_size, q_size, bias=True)

    def forward(self, x):
        z_mu, z_logvar = self.encoding(x)
        z_std = z_logvar.mul(0.5).exp()
        z_tilde = self.reparameterizing(z_mu, z_std)
        z_ = torch.tanh(self.fc_dec0(z_tilde))
        z_ = torch.tanh(self.fc_dec1(z_))
        x_mu = self.fc_dec(z_).view(-1, self.q_size)
        x_logvar = self.fc_dec2(z_).view(-1, self.q_size)
        # x_mu, x_logvar = self.decoding(z_)

        return x_mu, x_logvar, z_mu, z_logvar, z_tilde

    def encoding(self, x):
        z_1 = self.conv_1(x) #ReLU
        z_2 = self.conv_2(z_1) #ReLU
        z_3 = self.conv_3(z_2)  # ReLU
        z_4 = self.conv_4(z_3)  # ReLU
        z_5 = self.conv_5(z_4)  # ReLU

        z = torch.cat((z_1, z_2, z_3, z_4, z_5),dim=1).view(-1, self.q_size * 5*self.k)
        z = torch.relu(self.fc_z(z))
        z_ = torch.sigmoid(self.fc_z2(z))
        z_mu = self.fc_z_mu(z_)
        z_logvar = self.fc_z_logvar(z_)

        return z_mu, z_logvar

    # def decoding(self, z):
    #
    #     # z_ = self.deconv_1(z)
    #     # x_mu = self.deconv_2_1(z_)  #Tanh
    #     # x_logvar = self.deconv_2_1(z_)  #Tanh
    #     return x_mu, x_logvar

    def reparameterizing(self, mu, std):
        z = mu + std.mul(torch.randn_like(std))
        return z


class AE(nn.Module):
    def __init__(self, input_n, k, output_size, hidden_size):
        super(AE, self).__init__()
        self.output_size = output_size
        # encoding
        self.linear_1 = nn.Linear(input_n, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_8 = nn.Linear(hidden_size, hidden_size)
        self.linear_9 = nn.Linear(hidden_size, hidden_size)
        self.linear_10 = nn.Linear(input_n, input_n)
        # k
        self.linear_3 = nn.Linear(hidden_size, k)
        # decoding
        self.linear_4 = nn.Linear(k, hidden_size)
        self.linear_5 = nn.Linear(hidden_size, hidden_size)
        self.linear_7 = nn.Linear(hidden_size, hidden_size)
        self.linear_6 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        z = self.encoding(x)
        recon_x = self.decoding(z)
        return recon_x, z

    def encoding(self, x):
        x_ = torch.relu(self.linear_1(x))
        x_ = torch.relu(self.linear_2(x_))
        x_ = torch.relu(self.linear_8(x_))
        x_ = torch.tanh(self.linear_9(x_))
        z = self.linear_3(x_)
        return z

    def decoding(self, z):
        z_ = torch.relu(self.linear_4(z))
        z_ = torch.relu(self.linear_5(z_))
        z_ = torch.tanh(self.linear_7(z_))
        recon_x = self.linear_6(z_)
        return recon_x

# class HaarVAE(nn.Module):
#     def __init__(self, input_n, k, hidden_size, device):
#         super(HaarVAE, self).__init__()
#         self.device = device
#         self.hidden_size = hidden_size
#
#         # encoding
#         self.linear_1 = nn.Linear(input_n*2, hidden_size)
#         self.linear_2 = nn.Linear(hidden_size, hidden_size)
#         self.linear_3 = nn.Linear(hidden_size, hidden_size)
#
#         # mu & std
#         self.linear_1_ = nn.Linear(hidden_size, k)
#         self.linear_1__ = nn.Linear(hidden_size, k)
#
#         # decoding
#         self.linear_6 = nn.Linear(k, hidden_size)
#         self.linear_7 = nn.Linear(hidden_size, hidden_size)
#         self.linear_8 = nn.Linear(hidden_size, hidden_size)
#
#         self.linear_6_ = nn.Linear(hidden_size, input_n)
#         self.linear_6__ = nn.Linear(hidden_size, input_n)
#
#     def forward(self, x):
#         z_mu, z_logvar = self.encoding(x)
#
#         z = self.reparametrizing(z_mu, z_logvar)
#
#         x_mu, x_logvar = self.decoding(z)
#         return x_mu, x_logvar, z_mu, z_logvar, z
#
#     def encoding(self, x):
#         window_n = x.size(0)
#         x = torch.rfft(x,1,onesided=False)  # batch, window, 2
#         x = x.view(window_n,-1)
#         x = torch.tanh(self.linear_1(x))
#         x = torch.tanh(self.linear_2(x))
#         x = torch.tanh(self.linear_3(x))
#
#         z_mu = self.linear_1_(x)
#         z_logvar = self.linear_1__(x)
#
#         return z_mu, z_logvar
#
#     def reparametrizing(self, mu, log_var):
#         std = log_var.mul(0.5).exp()
#         eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).to(self.device)
#         z = std.mul(eps).add_(mu)
#         return z
#
#     def decoding(self, z):
#         z = torch.tanh(self.linear_6(z))
#         z = torch.tanh(self.linear_7(z))
#         z = torch.relu(self.linear_8(z))
#
#         x_mu = self.linear_6_(z)
#         x_logvar = self.linear_6__(z)
#
#         return x_mu, x_logvar
#
#     def wavelet_transform(self, x):
#         '''
#         :param x: Torch tensor, 2-dimensional time-series data (batch, window)
#         :return: V1, W1 (scaled data and wavelet transformed data)
#         '''
#         batch_size = x.size(0)
#         output_size = x.size(1)//2
#         s = 0.5 ** 0.5  #scale
#
#         # index for selecting odd, even column
#         ids_odd = torch.ByteTensor([1,0]).unsqueeze(0).repeat(batch_size,output_size)
#         ids_even = torch.ByteTensor([0,1]).unsqueeze(0).repeat(batch_size, output_size)
#
#         # window size of data is odd,
#         if x.size(1) % 2 != 0:
#             output_size += 1
#             ids_odd = torch.cat((ids_odd, torch.ByteTensor([1]*batch_size).view(-1,1)),dim=1)
#             ids_even = torch.cat((ids_even, torch.ByteTensor([1] * batch_size).view(-1, 1)), dim=1)
#
#         cAs = s*(x[ids_odd] + x[ids_even]).view(batch_size, output_size)
#         cDs = s * (x[ids_odd] - x[ids_even]).view(batch_size, output_size)
#
#         return cAs, cDs

# class HaarVAE(nn.Module):
#     def __init__(self, input_n, k, hidden_size, device):
#         super(HaarVAE, self).__init__()
#         self.device = device
#         self.hidden_size = hidden_size
#
#         # Haar Wavelet
#         self.wavelet_1 = nn.Linear(16, 16)
#         self.wavelet_2 = nn.Linear(24, 24)
#         self.wavelet_3 = nn.Linear(28, 28)
#         self.wavelet_4 = nn.Linear(30, 30)
#         self.wavelet_5 = nn.Linear(31, 31)
#
#         # encoding
#         self.linear_1 = nn.Linear(16, 16)
#         self.linear_2 = nn.Linear(24, 24)
#         self.linear_3 = nn.Linear(28, 28)
#         self.linear_4 = nn.Linear(30, 30)
#         self.linear_5 = nn.Linear(31, hidden_size)
#
#         # mu & std
#         self.linear_1_ = nn.Linear(hidden_size, k)
#         self.linear_1__ = nn.Linear(hidden_size, k)
#
#         # decoding
#         self.linear_6 = nn.Linear(k, hidden_size)
#         self.linear_7 = nn.Linear(hidden_size, hidden_size)
#         self.linear_8 = nn.Linear(hidden_size, hidden_size)
#
#         self.linear_6_ = nn.Linear(hidden_size, input_n)
#         self.linear_6__ = nn.Linear(hidden_size, input_n)
#
#     def forward(self, x):
#         z_mu, z_logvar = self.encoding(x)
#
#         z = self.reparametrizing(z_mu, z_logvar)
#
#         x_mu, x_logvar = self.decoding(z)
#         return x_mu, x_logvar, z_mu, z_logvar, z
#
#     def encoding(self, x):
#         v1, w1 = self.wavelet_transform(x)   # 32 -> 16
#         v2, w2 = self.wavelet_transform(v1)  # 16 -> 8
#         v3, w3 = self.wavelet_transform(v2)  # 8 -> 4
#         v4, w4 = self.wavelet_transform(v3)  # 4 -> 2
#         v5, w5 = self.wavelet_transform(v4)  # 2 -> 1
#
#         x1 = torch.tanh(self.wavelet_1(w1))
#         x1 = self.linear_1(x1)
#
#         x2 = torch.cat((x1, w2), dim=1)
#         x2 = torch.tanh(self.wavelet_2(x2))
#         x2 = self.linear_2(x2)
#
#         x3 = torch.cat((x2, w3), dim=1)
#         x3 = torch.tanh(self.wavelet_3(x3))
#         x3 = self.linear_3(x3)
#
#         x4 = torch.cat((x3, w4), dim=1)
#         x4 = torch.tanh(self.wavelet_4(x4))
#         x4 = self.linear_4(x4)
#
#         x5 = torch.cat((x4, v5), dim=1)
#         x5 = torch.relu(self.wavelet_5(x5))
#         x5 = torch.relu(self.linear_5(x5))
#
#         z_mu = self.linear_1_(x5)
#         z_logvar = self.linear_1__(x5)
#
#         return z_mu, z_logvar
#
#     def reparametrizing(self, mu, log_var):
#         std = log_var.mul(0.5).exp()
#         eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).to(self.device)
#         z = std.mul(eps).add_(mu)
#         return z
#
#     def decoding(self, z):
#         z = torch.tanh(self.linear_6(z))
#         z = torch.tanh(self.linear_7(z))
#         z = torch.relu(self.linear_8(z))
#
#         x_mu = self.linear_6_(z)
#         x_logvar = self.linear_6__(z)
#
#         return x_mu, x_logvar
#
#     def wavelet_transform(self, x):
#         '''
#         :param x: Torch tensor, 2-dimensional time-series data (batch, window)
#         :return: V1, W1 (scaled data and wavelet transformed data)
#         '''
#         batch_size = x.size(0)
#         output_size = x.size(1)//2
#         s = 0.5 ** 0.5  #scale
#
#         # index for selecting odd, even column
#         ids_odd = torch.ByteTensor([1,0]).unsqueeze(0).repeat(batch_size,output_size)
#         ids_even = torch.ByteTensor([0,1]).unsqueeze(0).repeat(batch_size, output_size)
#
#         # window size of data is odd,
#         if x.size(1) % 2 != 0:
#             output_size += 1
#             ids_odd = torch.cat((ids_odd, torch.ByteTensor([1]*batch_size).view(-1,1)),dim=1)
#             ids_even = torch.cat((ids_even, torch.ByteTensor([1] * batch_size).view(-1, 1)), dim=1)
#
#         cAs = s*(x[ids_odd] + x[ids_even]).view(batch_size, output_size)
#         cDs = s * (x[ids_odd] - x[ids_even]).view(batch_size, output_size)
#
#         return cAs, cDs

class HaarVAE(nn.Module):
    def __init__(self, input_n, k, hidden_size, device):
        super(HaarVAE, self).__init__()
        self.device = device
        self.hidden_size = hidden_size


        # Haar Wavelet
        output_dim = self.get_dim(input_n, arc_len=5)
        self.wavelet_1 = nn.Linear(output_dim[0], hidden_size)
        self.wavelet_2 = nn.Linear(output_dim[1], hidden_size)
        self.wavelet_3 = nn.Linear(output_dim[2], hidden_size)
        self.wavelet_4 = nn.Linear(output_dim[3], hidden_size)
        self.wavelet_5 = nn.Linear(output_dim[3], hidden_size)

        # encoding
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, hidden_size)
        self.linear_5 = nn.Linear(hidden_size, hidden_size)
        # mu & std
        self.linear_1_ = nn.Linear(hidden_size, 1)
        self.linear_2_ = nn.Linear(hidden_size, 1)
        self.linear_3_ = nn.Linear(hidden_size, 1)
        self.linear_4_ = nn.Linear(hidden_size, 1)
        self.linear_5_ = nn.Linear(hidden_size, 1)

        self.linear_1__ = nn.Linear(hidden_size, 1)
        self.linear_2__ = nn.Linear(hidden_size, 1)
        self.linear_3__ = nn.Linear(hidden_size, 1)
        self.linear_4__ = nn.Linear(hidden_size, 1)
        self.linear_5__ = nn.Linear(hidden_size, 1)
        # decoding
        self.linear_6 = nn.Linear(1, hidden_size)
        self.linear_7 = nn.Linear(1, hidden_size)
        self.linear_8 = nn.Linear(1, hidden_size)
        self.linear_9 = nn.Linear(1, hidden_size)
        self.linear_10 = nn.Linear(1, hidden_size)

        self.linear_6_ = nn.Linear(hidden_size, input_n)
        self.linear_7_ = nn.Linear(hidden_size, input_n)
        self.linear_8_ = nn.Linear(hidden_size, input_n)
        self.linear_9_ = nn.Linear(hidden_size, input_n)
        self.linear_10_ = nn.Linear(hidden_size, input_n)

        self.linear_6__ = nn.Linear(hidden_size, input_n)
        self.linear_7__ = nn.Linear(hidden_size, input_n)
        self.linear_8__ = nn.Linear(hidden_size, input_n)
        self.linear_9__ = nn.Linear(hidden_size, input_n)
        self.linear_10__ = nn.Linear(hidden_size, input_n)

        self.decoder_mu = nn.Linear(input_n*5, input_n)
        self.decoder_logvar = nn.Linear(input_n * 5, input_n)

    def forward(self, x):
        z_mu, z_logvar = self.encoding(x)

        z = self.reparametrizing(z_mu, z_logvar)

        x_mu, x_logvar = self.decoding(z)
        return x_mu, x_logvar, z_mu, z_logvar, z

    def encoding(self, x):
        v1, w1 = self.wavelet_transform(x)   # 1440 -> 720
        v2, w2 = self.wavelet_transform(v1)  # 720 -> 360
        v3, w3 = self.wavelet_transform(v2)  # 360 -> 180
        v4, w4 = self.wavelet_transform(v3)  # 180 -> 90
        # v5, w5 = self.wavelet_transform(v4)  # 90 -> 45

        x1 = torch.tanh(self.wavelet_1(w1))
        x1 = torch.tanh(self.linear_1(x1))
        z1_mu = self.linear_1_(x1)
        z1_logvar = self.linear_1__(x1)

        x2 = torch.tanh(self.wavelet_2(w2))
        x2 = torch.tanh(self.linear_2(x2))
        z2_mu = self.linear_2_(x2)
        z2_logvar = self.linear_2__(x2)

        x3 = torch.tanh(self.wavelet_3(w3))
        x3 = torch.tanh(self.linear_3(x3))
        z3_mu = self.linear_3_(x3)
        z3_logvar = self.linear_3__(x3)

        x4 = torch.tanh(self.wavelet_4(w4))
        x4 = torch.tanh(self.linear_4(x4))
        z4_mu = self.linear_4_(x4)
        z4_logvar = self.linear_4__(x4)

        x5 = torch.relu(self.wavelet_5(v4))
        x5 = torch.relu(self.linear_5(x5))
        z5_mu = self.linear_5_(x5)
        z5_logvar = self.linear_5__(x5)

        z_mu = torch.cat((z1_mu,z2_mu,z3_mu,z4_mu,z5_mu), dim=1)
        z_logvar = torch.cat((z1_logvar, z2_logvar, z3_logvar, z4_logvar, z5_logvar), dim=1)
        return z_mu, z_logvar

    def reparametrizing(self, mu, log_var):
        std = log_var.mul(0.5).exp()
        eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).to(self.device)
        z = std.mul(eps).add_(mu)
        return z

    def decoding(self, z):
        z1, z2, z3, z4, z5 = z[:,0].unsqueeze(-1), z[:,1].unsqueeze(-1),z[:,2].unsqueeze(-1),z[:,3].unsqueeze(-1),z[:,4].unsqueeze(-1)

        z1 = torch.tanh(self.linear_6(z1))
        x1_mu = self.linear_6_(z1)
        x1_logvar = self.linear_6__(z1)

        z2 = torch.tanh(self.linear_7(z2))
        x2_mu = self.linear_7_(z2)
        x2_logvar = self.linear_7__(z2)

        z3 = torch.tanh(self.linear_8(z3))
        x3_mu = self.linear_8_(z3)
        x3_logvar = self.linear_8__(z3)

        z4 = torch.tanh(self.linear_9(z4))
        x4_mu = self.linear_9_(z4)
        x4_logvar = self.linear_9__(z4)

        z5 = torch.relu(self.linear_10(z5))
        x5_mu = self.linear_10_(z5)
        x5_logvar = self.linear_10__(z5)

        x_mu = x1_mu + x2_mu + x3_mu + x4_mu + x5_mu
        x_logvar = x1_logvar + x2_logvar + x3_logvar + x4_logvar + x5_logvar

        # x_mu = self.decoder_mu(torch.cat((x1_mu, x2_mu, x3_mu, x4_mu, x5_mu), dim=1))
        # x_logvar = self.decoder_logvar(torch.cat((x1_logvar, x2_logvar, x3_logvar, x4_logvar, x5_logvar), dim=1))
        return x_mu, x_logvar

    def wavelet_transform(self, x):
        '''
        :param x: Torch tensor, 2-dimensional time-series data (batch, window)
        :return: V1, W1 (scaled data and wavelet transformed data)
        '''
        batch_size = x.size(0)
        output_size = x.size(1)//2
        s = 0.5 ** 0.5  #scale

        # index for selecting odd, even column
        ids_odd = torch.ByteTensor([1,0]).unsqueeze(0).repeat(batch_size,output_size)
        ids_even = torch.ByteTensor([0,1]).unsqueeze(0).repeat(batch_size, output_size)

        # window size of data is odd,
        if x.size(1) % 2 != 0:
            output_size += 1
            ids_odd = torch.cat((ids_odd, torch.ByteTensor([1]*batch_size).view(-1,1)),dim=1)
            ids_even = torch.cat((ids_even, torch.ByteTensor([1] * batch_size).view(-1, 1)), dim=1)

        cAs = s*(x[ids_odd] + x[ids_even]).view(batch_size, output_size)
        cDs = s * (x[ids_odd] - x[ids_even]).view(batch_size, output_size)

        return cAs, cDs

    def get_dim(self, data_len, arc_len):
        output_dim = np.zeros(arc_len + 1, dtype=int)
        output_dim[0] = data_len
        for i in range(1, arc_len + 1):
            if output_dim[i-1] % 2 == 0:
                output_dim[i] = output_dim[i-1] // 2
            else:
                 output_dim[i] = (output_dim[i-1] // 2) +1
        return output_dim[1:arc_len+1]

# class HaarVAE(nn.Module):
#     def __init__(self, input_n, k, hidden_size, device):
#         super(HaarVAE, self).__init__()
#         self.device = device
#         self.hidden_size = hidden_size
#
#         # Haar Wavelet
#         self.wavelet_1 = nn.Linear(16, hidden_size)
#         self.wavelet_2 = nn.Linear(8, hidden_size)
#         self.wavelet_3 = nn.Linear(4, hidden_size)
#         self.wavelet_4 = nn.Linear(4, hidden_size)
#
#         # encoding
#         self.linear_1 = nn.Linear(hidden_size*3, hidden_size)
#         self.linear_2 = nn.Linear(hidden_size, hidden_size)
#         self.linear_3 = nn.Linear(hidden_size, hidden_size)
#         # mu & std
#         self.linear_31 = nn.Linear(hidden_size, k)
#         self.linear_32 = nn.Linear(hidden_size, k)
#         # decoding
#         self.linear_6 = nn.Linear(k, hidden_size)
#         self.linear_7 = nn.Linear(hidden_size, hidden_size)
#         self.linear_8 = nn.Linear(hidden_size, hidden_size)
#         self.linear_9 = nn.Linear(hidden_size, hidden_size)
#         self.linear_10 = nn.Linear(hidden_size, hidden_size)
#
#         self.linear_61 = nn.Linear(hidden_size, input_n)
#         self.linear_62 = nn.Linear(hidden_size, input_n)
#
#     def forward(self, x):
#         v1, w1 = self.wavelet_transform(x)
#         v2, w2 = self.wavelet_transform(v1)
#         v3, w3 = self.wavelet_transform(v2)
#
#         w1 = torch.tanh(self.wavelet_1(w1))
#         w2 = torch.tanh(self.wavelet_2(w2))
#         w3 = torch.tanh(self.wavelet_3(w3))
#
#         v3, _ = self.wavelet_transform(v3)  # 4 -> 2
#         v3, _ = self.wavelet_transform(v3)  # 2 -> 1 (scalar)
#
#         x1 = torch.cat((w1,w2,w3), dim=1)
#
#         x1 = x1.view(-1, self.hidden_size*3)
#         z_mu, z_logvar = self.encoding(x1)
#         z = self.reparametrizing(z_mu, z_logvar)
#         x_mu, x_logvar = self.decoding(z)
#
#         bias = (v3*((0.5 ** 0.5)**5)).repeat(1,x.size(-1))
#         x_mu += bias
#         return x_mu, x_logvar, z_mu, z_logvar, z
#
#     def encoding(self, x):
#         x_ = torch.tanh(self.linear_1(x))
#         x_ = torch.tanh(self.linear_2(x_))
#         x_ = torch.tanh(self.linear_3(x_))
#         z_mu = self.linear_31(x_)
#         z_logvar = self.linear_32(x_)
#         return z_mu, z_logvar
#
#     def reparametrizing(self, mu, log_var):
#         std = log_var.mul(0.5).exp()
#         eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).to(self.device)
#         z = std.mul(eps).add_(mu)
#         return z
#
#     def decoding(self, z):
#         z_ = torch.tanh(self.linear_6(z))
#         z_ = torch.tanh(self.linear_7(z_))
#         z_ = torch.tanh(self.linear_8(z_))
#         x_mu = self.linear_61(z_)
#         x_logvar = (self.linear_62(z_))
#         return x_mu, x_logvar
#
#     def wavelet_transform(self, x):
#         '''
#         :param x: Torch tensor, 2-dimensional time-series data (batch, window)
#         :return: V1, W1 (scaled data and wavelet transformed data)
#         '''
#         batch_size = x.size(0)
#         output_size = x.size(1)//2
#         s = 0.5 ** 0.5  #scale
#
#         # index for selecting odd, even column
#         ids_odd = torch.ByteTensor([1,0]).unsqueeze(0).repeat(batch_size,output_size)
#         ids_even = torch.ByteTensor([0,1]).unsqueeze(0).repeat(batch_size, output_size)
#
#         # window size of data is odd,
#         if x.size(1) % 2 != 0:
#             output_size += 1
#             ids_odd = torch.cat((ids_odd, torch.ByteTensor([1]*batch_size).view(-1,1)),dim=1)
#             ids_even = torch.cat((ids_even, torch.ByteTensor([1] * batch_size).view(-1, 1)), dim=1)
#
#         cAs = s*(x[ids_odd] + x[ids_even]).view(batch_size, output_size)
#         cDs = s * (x[ids_odd] - x[ids_even]).view(batch_size, output_size)
#
#         return cAs, cDs















