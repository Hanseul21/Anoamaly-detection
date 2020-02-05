import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from model import Conv2AE, WAE, mmd, sampling
import os
import numpy as np
batch_size = 256
hidden_size = 2
lr = 0.001
cls_ = 7
ratio = 0.03
# data loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])
dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# trainset = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
# selection of specific number (like 0,1,2,...)
ids = dataset.targets==cls_
dataset.data = dataset.data[ids]
dataset.targets = dataset.targets[ids]
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

def visualize_z_space(z, y=None, loc = None, e=None,cls=7):
    if type(z).__module__ == torch.__name__:
        z = z.detach().cpu().numpy()
    # if type(y).__module__ == torch.__name__:
    #     y = y.detach().cpu().numpy()

    plt.figure()
    z_x, z_y = z[:,0], z[:,1]
    # c_ = ['green' if label == cls else 'red' for label in y]
    if y is None:
        c_ = ['green' if i > 6 else 'red' for i in range(len(z))]
    else:
        c_ = ['green' if label == 0 else 'red' for label in y]
    plt.scatter(z_x, z_y, c=c_)
    plt.savefig(os.path.join(loc,'{0} result of z space.png'.format(e)))

def training(net_type, epoch=1000, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if net_type in ['Conv2AE']:
        net = Conv2AE(hidden_size=2).to(device)
    elif net_type in ['WAE', 'wae']:
        net = WAE(hidden_size=2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for e in range(epoch):
        valid = 0
        plot_z = None
        plot_label = None
        for b, data in enumerate(trainloader):
            input, target = data
            x, y = Variable(input).to(device), Variable(target).to(device)
            idx_rand = int(x.size(0)*ratio)
            new_size = [idx_rand]
            new_size.extend(list(x.size()[1:]))
            noise = torch.rand(new_size).to(device)
            x_ = x
            x_[:idx_rand] += noise
            recon, z_tilde = net(x_)
            # recon, z_mean, z_log_var = net(x_)

            z = sampling(1,template=z_tilde)
            optimizer.zero_grad()

            mmd_loss = mmd(z_tilde,z,1)
            recon_loss = (recon - x_).abs_().sum().div(x.size(0))
            loss = recon_loss + mmd_loss
            # loss = criterion(recon, x_)
            valid += loss
            loss.backward()
            optimizer.step()
            if plot_z is None:
                plot_z = z_tilde
                plot_label = [0 if i >= idx_rand else 1 for i in range(x.size(0))]
            else:
                plot_z = torch.cat((plot_z, z_tilde), dim=0)
                plot_label.extend([0 if i >= idx_rand else 1 for i in range(x.size(0))])

            if (b+1) % 5 == 0:
                print('loss ',valid)
                valid = 0
        if plot and (e+1) % 100 == 0:
            loc = 'plot'
            visualize_z_space(plot_z,plot_label, loc = loc, e=(e+1)/100)
            fig = plt.figure()
            fig.add_subplot(2,2,1)
            plt.imshow(x[0].squeeze().detach().cpu().numpy())
            plt.xlabel('noised original')
            fig.add_subplot(2,2,2)
            plt.imshow(recon[0].squeeze().detach().cpu().numpy())
            plt.xlabel('noised recon')
            fig.add_subplot(2,2,3)
            plt.imshow(x[8].squeeze().detach().cpu().numpy())
            plt.xlabel('clear original')
            fig.add_subplot(2,2,4)
            plt.imshow(recon[8].squeeze().detach().cpu().numpy())
            plt.xlabel('clear recon')
            # plt.show()
            plt.savefig(os.path.join(loc,'{0} result of image.png'.format((e+1)/100)))

training('WAE')
