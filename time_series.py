import torch
from VAE import VAENet, weight_init
from WAE import WAENet, kaiming_init, mmd, im_kernel_sum
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from Evaluation import evaluate
from data_ import KPI, NAB, Yahoo
import time
import pandas as pd
import os
import numpy as np

# VAE setting
epoch = 50
q_size = 60
input_n = q_size
hidden_n = 10
output_size = q_size
k = 2
lr = 0.001

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

# threshold = 0.3
# L = 5   # monte-carlo method

#
# train_dataloaders = KPI('train', norm=True, q_size=1440, batch_size=256, ratio=0.7)
# test_dataloaders= KPI('test', norm=True, q_size=1440, batch_size=256, ratio=0.7)

# test_book = ['artificialWithAnomaly','realAdExchange','realAWSCloudwatch','realKnownCause','realTraffic','realTweets']
# train_dataloaders = NAB('train', dir='realKnownCause', norm=True, q_size=128, batch_size=64, ratio=0.7, shuffle=False)
# test_dataloaders = NAB('test', dir='realKnownCause', norm=True, q_size=128, batch_size=64, ratio=0.7, shuffle=False)

# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


train_dataloaders = Yahoo('train', dir='A3Benchmark', norm=True, q_size=60, batch_size=256, ratio=0.7)
test_dataloaders = Yahoo('test', dir='A3Benchmark', norm=True, q_size=60, batch_size=256, ratio=0.7)
data_type ='yahoo'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for i, train_loader in enumerate(train_dataloaders):
    evaluator = evaluate(data_type)
    batch_n = len(train_loader)

    net = VAENet(input_n, hidden_n, k, output_size, device).to(device)
    net.apply(weight_init)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    for e in range(epoch):
        valid = 0
        for b, data in enumerate(train_loader):
            x, y = data['value'].to(device), data['label'].to(device)
            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()

            x_mu, x_logvar, z_mu, z_logvar = net(x)
            x = x.view(-1,output_size)

            KLD = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

            x_std = x_logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(x_std.size()).normal_(mean=0, std=1).to(device)
            recon_x = eps.mul(x_std).add(x_mu)
            y_raw = evaluator.get_log_prob(x, x_mu, x_std)
            # recon_loss = y_raw.mean()

            recon_loss = criterion(recon_x, x)
            loss = recon_loss + KLD
            valid += loss
            loss.backward()
            optimizer.step()
        print('loss : {0:.4f}'.format(valid))
        # print(recon_loss)
        # print(KLD)
        # print(y_raw.mean())
            # if (b + 1)% 10 == 0:
            #     true = (y != 0).sum(dim=1).sum(dim=0).item()
            #     if true != 0:
            #         _, recall, prec, th, f1 = evaluator.pr_auc(y_raw, y, plot=False)
            #         argmax = np.argmax(f1)
            #         y_pred = evaluator.get_score(y_raw, th[argmax])
            #         new_pred = evaluator.anomaly_seg(y_pred, y)
            #         # f1 = evaluator.f1(y_pred, y)
            #         f1_corr = evaluator.f1(new_pred, y)
            #
            #         print('Best f1-score {0:.5f} corrected f1-score {1:.5f}'.format(np.max(f1), f1_corr))
            #         print('recall {0:.5f} precision {1:.5f}'.format(recall[argmax], prec[argmax]))
            #         print('true', true)
            #     else:
            #         print('nothing')

    for i_test, test_loader in enumerate(test_dataloaders):
        for b, data in enumerate(test_loader):
            x, y = data['value'].to(device), data['label'].to(device)
            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()

            x_mu, x_logvar, z_mu, z_logvar = net(x)
            x_std = x_logvar.mul(0.5).exp_()
            x = x.view(-1, output_size)
            recon_x = sampling(mu=x_mu, sigma=x_std)

            y_raw = evaluator.get_log_prob(x, x_mu, x_std)
            true = (y != 0).sum(dim=1).sum(dim=0).item()
            if true == 0:
                recall, prec, th, f1 = [0],[0],[0],[0]
                argmax = 0
            else:
                _, recall, prec, th, f1 = evaluator.pr_auc(y_raw, y)
                # print(e*batch_n + b)
                # print(y_raw)
                # print('recall   :', recall)
                # print('prec     :', prec)
                # print('threshold:', th)
                # print('f1 score :', f1)
                argmax = np.argmax(f1)
                y_pred = evaluator.get_score(y_raw, th[argmax])
                new_pred = evaluator.anomaly_seg(y_pred, y)
                # f1 = evaluator.f1(y_pred, y)
                f1_corr = evaluator.f1(new_pred, y)
                print('Best f1-score {0:.5f} corrected f1-score {1:.5f}'.format(np.max(f1), f1_corr))
                print('recall {0:.5f} precision {1:.5f}'.format(recall[argmax], prec[argmax]))
                print('true', true)
                evaluator.record(e * batch_n + b, recall[argmax], prec[argmax], f1[argmax], true, th[argmax])
                print('recorded')

                recon_prob = evaluator.get_log_prob(recon_x, mu=x_mu, std=x_std)

                _, recall_prob, prec_prob, th_prob, f1_prob = evaluator.pr_auc(recon_prob, y)
                argmax_ = np.argmax(f1_prob)
                recall_prob, prec_prob, th_prob, f1_prob = \
                    recall_prob[argmax_], prec_prob[argmax_], th_prob[argmax_], f1_prob[argmax_]

                print('reconstruction probability')
                print('recall :{0:.4f}, precision :{1:.4f}, threshold :{2:.4f},'
                      ' f1 score :{3:.4f}'.format(recall_prob, prec_prob, th_prob, f1_prob))
                break
            break
        evaluator.get_record('normal VAE training record {0}'.format(i+1))

print('train is the end')
# for data in test:
#     print(test)