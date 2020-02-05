import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


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
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x

class AE(nn.Module):
    def __init__(self, input_n, hidden_n, k, output_size, device):
        super(AE, self).__init__()
        self.device = device
        self.output_size = output_size
        # encoding
        self.fc1 = nn.Linear(input_n, hidden_n)
        # mu & std
        self.fc2 = nn.Linear(hidden_n, k)
        # decoding
        self.fc3 = nn.Linear(k, hidden_n)
        self.fc4 = nn.Linear(hidden_n, output_size)

    def forward(self, x):
        z = self.encoding(x)
        recon_x = self.decoding(z)

        return x, recon_x

    def encoding(self, x):
        x = F.relu(self.fc1(x))
        z = F.relu(self.fc2(x))
        return z

    def decoding(self, z):
        recon_x = F.relu(self.fc3(z))
        recon_x = self.fc4(recon_x)

        return recon_x


class Conv2AE(nn.Module):
    def __init__(self, hidden_size):
        super(Conv2AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),    #  in_channels, out_channels, kernel_size, stride, padding, diction , groups, bias
                                                        # B,  64, 14, 14
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # B,  128, 7, 7
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # B, 256,  3,  3
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            View((-1, 128 * 3 * 3)),  # B, 256*4*4
            nn.Linear(128 * 3 * 3, hidden_size)  # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128 * 7 * 7),  # B, 256*4*4
            View((-1, 128, 7, 7)),  # B, 256,  4,  4
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # B,  64, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # B,  32, 28, 28
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 1),  # B, nc, 28, 28
        )

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)

        return x, z

class WAE(nn.Module):
    def __init__(self, hidden_size, clipping=True):
        super(WAE, self).__init__()
        self.clipping = clipping
        # encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),  # B, 1, 28, 28
            # in_channels, out_channels, kernel_size, stride, padding, diction , groups, bias
            # B,  64, 14, 14
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # B, 32, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # B, 64, 7, 7
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            View((-1, 128 * 3 * 3)),                  # B, 128*3*3
            nn.Linear(128 * 3 * 3, hidden_size)       # B, z_dim
        )
        #
        # self.linear1 =
        # self.linear2 = nn.Linear(128 * 3 * 3, hidden_size)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128 * 7 * 7),    # B, 128*3*3
            View((-1, 128, 7, 7)),                  # B, 128,  7,  7
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # B,  64, 14, 14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # B,  32, 28, 28
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 1),  # B, nc, 28, 28
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, decoding=False):
        if decoding:
            # input is z
            z_mean = None
            z_var = None
            z = x
        else:
            z_tilde = self.encoder(x)
            # = self.linear1(z), self.linear2(z)
            # z_var = z_log_var.exp_()
            # if self.clipping:
            #     z_var = torch.clamp(z_var, -0.5, 0.5)
            # z = self.reparameterizing(z_mean, z_var)
        # x_mu, x_logvar = self.decoder(z)
        recon = self.decoder(z_tilde)
        # z_log_var = z_var.log_()

        # return x_mu, x_logvar, z_mean, z_log_var
        return recon, z_tilde
    #
    # def encoding(self, x):
    #     x = F.relu(self.fc1(x))
    #     z_mu = F.relu(self.fc21(x))
    #     z_log_var = F.relu(self.fc22(x))
    #     return z_mu, z_log_var
    #
    # def decoding(self, z):
    #     recon_x = F.relu(self.fc3(z))
    #     x_mu = F.relu(self.fc41(recon_x))
    #     x_logvar = F.relu(self.fc42(recon_x))
    #     return x_mu, x_logvar

    def reparameterizing(self, mu, z_var):
        z = mu + torch.randn_like(mu).to('cuda').mul((1e-8 + z_var).sqrt())
        return z

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def mmd(z_tilde, z, z_var):
    r"""Calculate maximum mean discrepancy described in the WAE paper.
    Args:
        z_tilde (Tensor): samples from deterministic non-random encoder Q(Z|X).
            2D Tensor(batch_size x dimension).
        z (Tensor): samples from prior distributions. same shape with z_tilde.
        z_var (Number): scalar variance of isotropic gaussian prior P(Z).
    """
    assert z_tilde.size() == z.size(), print(z_tilde.size(), ' ' ,z.size())
    assert z.ndimension() == 2

    n = z.size(0)
    out = im_kernel_sum(z, z, z_var, exclude_diag=True).div(n*(n-1)) + \
          im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n*(n-1)) + \
          -im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n*n).mul(2)

    return out

def im_kernel_sum(z1, z2, z_var, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum

def sampling(sigma, mu = None, template=None, size=None, device='cuda'):
    '''
    sampling z drawn from N(mean is 0, standard deviation is 1)
    :param n_sample: batch_size
    :param dim: hidden dimension
    :param sigma: learned std
    :param template: ?
    :return: sampled z
    '''
    # if type(sigma).__module__ != torch.__name__:
    #     sigma = torch.FloatTensor(size).fill_(sigma).to(device)

    if mu is None:
        mu = torch.zeros_like(template)

    if template is not None:
        z = mu + sigma * Variable(template.data.new(template.size()).normal_())
    else:
        z = mu + sigma * torch.randn_like(sigma).to(device)
        z = Variable(z)

    return z
