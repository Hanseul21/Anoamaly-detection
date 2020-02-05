import torch.nn as nn
import torch
import torch.nn.functional as F

class VAENet(nn.Module):
    def __init__(self, input_n, hidden_n, k, output_size, device):
        super(VAENet, self).__init__()
        self.device = device
        self.output_size = output_size
        # encoding
        self.fc1 = nn.Linear(input_n, hidden_n)
        # mu & std
        self.fc21 = nn.Linear(hidden_n, k)
        self.fc22 = nn.Linear(hidden_n, k)
        # decoding
        self.fc3 = nn.Linear(k, hidden_n)
        self.fc41 = nn.Linear(hidden_n, output_size)
        self.fc42 = nn.Linear(hidden_n, output_size)

    def forward(self, x):
        z_mu, z_logvar = self.encoding(x)
        z = self.reparametrizing(z_mu, z_logvar)
        x_mu, x_logvar = self.decoding(z)

        return x_mu, x_logvar, z_mu, z_logvar

    def encoding(self, x):
        x = F.relu(self.fc1(x))
        mu = F.relu(self.fc21(x))
        log_var = F.relu(self.fc22(x))
        return mu, log_var

    def reparametrizing(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).to(self.device)
        z = eps.mul(std).add_(mu)
        return z

    def decoding(self, z):
        recon_x = F.relu(self.fc3(z))
        x_mu = self.fc41(recon_x)
        x_logvar = self.fc42(recon_x)

        # recon_x = F.sigmoid(self.fc4(recon_x))
        return x_mu, x_logvar

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('fc') != -1:
        torch.nn.init.kaiming_normal(m.weight.data)
        torch.nn.init.kaiming_normal(m.bias.data)
