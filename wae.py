import torch
import torch.nn as nn
from torch.autograd import Variable
from WAE import WAENet, mmd, sample_z
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import autograd
import math

import numpy as np

# data loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])
trainset = dsets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = dsets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

print('Data configuration')
print('Train :', trainset.train_data.size())
print('Test :', testset.test_data.size())

# setting
input_n = 28*28
hidden_n = 10
k = 10

n_test = 8
n_epoch = 20
n_valid = 100
batch_size = 100
lr = 0.001

z_var = 0.5
clipping = False
grad_reg = 0

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=n_test, shuffle=True, num_workers=2 )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = WAENet(z_dim=hidden_n, nc=1, clipping=clipping).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.BCELoss(size_average=False)


for e in range(n_epoch):
    valid_loss = 0.0
    for b, data in enumerate(trainloader):
        image, label = data
        image, label = Variable(image).cuda(), Variable(label).cuda()

        optimizer.zero_grad()

        # recon loss
        x_recon, z_mean, z_var = net(image)

        z = sample_z(batch_size, hidden_n, z_var)
        z_tilda = z_mean + torch.randn(z_mean.size(0),hidden_n).to(device).mul((1e-8 + z_var.exp()).sqrt())
        # mmd loss
        MMD = mmd(z_tilda, z, z_var=z_var)
        recon_loss = F.mse_loss(x_recon, image, size_average=False).div(batch_size)
        if clipping is False:
            eps = torch.randn(z_mean.size(0),hidden_n).to(device)
            z_hat = eps*z + (1-eps)*z_tilda
            z_output, _, __ = net(z_hat,True)
            z_d = autograd.grad(outputs=z_output, inputs=z_hat, grad_outputs=torch.ones(z_output.size()).to(device))[0]
            grad_reg = ((z_d.norm(2) - 1)**2).div(batch_size)
        loss = recon_loss + MMD + grad_reg
        loss.backward()
        optimizer.step()

        valid_loss += loss
        if (b % n_valid) == (n_valid - 1):
            print('{0:2d} epoch {1:4d} batch : loss : {2:.5f}'.format(e, b + 1, valid_loss / n_valid))
            valid_loss = 0.0

fig = plt.figure()

test_input, label = next(iter(testloader))
test_input = test_input
test_input = Variable(test_input)
test_output, z_mean, z_var = net(test_input.to(device))

row = n_test
col = 2

for r in range(row):
    test_input = test_input.view((test_input.size(0), 28, 28))
    test_output = test_output.view((test_output.size(0), 28, 28))
    ax = fig.add_subplot(row, col, 2*r+1)
    ax.imshow(test_input[r].numpy())
    ax.set_xlabel('input')

    bx=fig.add_subplot(row, col, 2*r+2)
    bx.imshow(test_output[r].cpu().detach().numpy())
    bx.set_xlabel('output')
plt.title('WAE')
plt.show()
