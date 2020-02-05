import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

class WAE(nn.Module):
    def __init__(self, input_n, hidden_n, k, output_size, device, clipping=True):
        super(WAE, self).__init__()
        self.clipping = clipping
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
            z_mean, z_log_var = self.encoding(x)
            z_var = z_log_var.exp_()
            if self.clipping:
                z_var = torch.clamp(z_var, -0.5, 0.5)
            z = self.reparameterizing(z_mean, z_var)
        x_mu, x_logvar = self.decoding(z)
        z_log_var = z_var.log_()

        return x_mu, x_logvar, z_mean, z_log_var

    def encoding(self, x):
        x = F.relu(self.fc1(x))
        z_mu = F.relu(self.fc21(x))
        z_log_var = F.relu(self.fc22(x))
        return z_mu, z_log_var

    def decoding(self, z):
        recon_x = F.relu(self.fc3(z))
        x_mu = F.relu(self.fc41(recon_x))
        x_logvar = F.relu(self.fc42(recon_x))
        return x_mu, x_logvar

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
    if type(sigma).__module__ != torch.__name__:
        sigma = torch.FloatTensor(size).fill_(sigma).to(device)

    if mu is None:
        mu = torch.zeros_like(sigma)

    if template is not None:
        z = mu + sigma * Variable(template.data.new(template.size()).normal_())
    else:
        z = mu + sigma * torch.randn_like(sigma).to(device)
        z = Variable(z)

    return z