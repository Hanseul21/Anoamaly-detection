import torch
from VAE import VAENet, weight_init
from WAE import WAE, kaiming_init, mmd, im_kernel_sum, sampling
import torch.nn.functional as F
import torch.autograd as autograd
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
epoch = 100
q_size = 128
batch_size = 256
z_var = 1
z_std = np.sqrt(z_var)
clipping = True
input_n = q_size
hidden_n = 10
output_size = q_size
k = 2
lr = 0.0001
train_dataloaders = Yahoo('train', dir='A3Benchmark', norm=True, q_size=q_size, batch_size=batch_size, ratio=0.7)
test_dataloaders = Yahoo('test', dir='A3Benchmark', norm=True, q_size=q_size, batch_size=batch_size, ratio=0.7)
data_type ='yahoo'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = WAE(input_n, hidden_n, k, output_size, device, clipping).to(device)
net.apply(weight_init)
optimizer = optim.Adam(net.parameters(), lr=lr)



    # criterion = nn.MSELoss(reduction='mean')
for i, train_loader in enumerate(train_dataloaders):
    evaluator = evaluate(data_type)
    for e in range(epoch):
        valid = 0
        for b, data in enumerate(train_loader):
            # batch_n = len(train_loader)
            x, y = data['value'].to(device), data['label'].to(device)
            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()

            x_mu, x_logvar, z_mean_, z_logvar_ = net(x)
            x_std = x_logvar.mul(0.5).exp_()
            recon_x = sampling(mu=x_mu, sigma=x_std)
            z_std_ = z_logvar_.exp_().sqrt()

            z = sampling(sigma=z_std, size=(z_mean_.size()))
            z_tilde = z_mean_ + (z_std_.exp() + 1e-8).sqrt() * torch.randn(z_mean_.size()).to(device)

            # mmd loss
            MMD = mmd(z_tilde, z, z_var=z_var)

            recon_loss = F.mse_loss(recon_x, x, size_average=False).div(batch_size)
            if clipping is False:
                eps = torch.randn(z_mean_.size(0), k).to(device)
                z_hat = eps * z + (1 - eps) * z_tilde
                z_output, _, __ = net(z_hat, True)
                z_d = \
                autograd.grad(outputs=z_output, inputs=z_hat, grad_outputs=torch.ones(z_output.size()).to(device))[0]
                grad_reg = ((z_d.norm(2) - 1) ** 2).div(batch_size)
            else:
                grad_reg = 0
            loss = recon_loss + MMD + grad_reg
            valid += loss.item()

            loss.backward()
            optimizer.step()
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ loss {0}'.format(valid))
        print('reconstruction loss')
        evaluator.get_result((recon_x - x).abs_(), y)
        print('reconstruction probability')
        recon_prob = evaluator.get_log_prob(x,x_mu,x_std)
        evaluator.get_result(recon_prob, y)

#         if e % 10 == 0:
#             for valid_loader in test_dataloaders:
#                 for i, data in enumerate(valid_loader):
#                     x, y = data['value'].to(device), data['label'].to(device)
#                     x, y = Variable(x), Variable(y)
#
#                     optimizer.zero_grad()
#
#                     x_mu, x_logvar, z_mean_, z_logvar_ = net(x)
#                     print(x_logvar)
#                     x_std = x_logvar.mul(0.5).exp_()
#                     recon_x = sampling(mu=x_mu, sigma=x_std)
#                     z_std_ = z_logvar_.exp_().sqrt()
#
#                     z = sampling(sigma=z_std, size=(z_mean_.size()))
#                     z_tilde = z_mean_ + (z_std_.exp() + 1e-8).sqrt() * torch.randn(z_mean_.size()).to(device)
#
#                     # mmd loss
#                     MMD = mmd(z_tilde, z, z_var=z_var)
#
#                     recon_loss = F.mse_loss(recon_x, x, size_average=False).div(batch_size)
#                     if clipping is False:
#                         eps = torch.randn_like(z_mean_).to(device)
#                         z_hat = eps * z + (1 - eps) * z_tilde
#                         z_output, _, __ = net(z_hat, True)
#                         z_d = \
#                             autograd.grad(outputs=z_output, inputs=z_hat, grad_outputs=torch.ones(z_output.size()).to(device))[
#                                 0]
#                         grad_reg = ((z_d.norm(2) - 1) ** 2).div(batch_size)
#                     else:
#                         grad_reg = 0
#
#                     loss = recon_loss + MMD + grad_reg
#                     recon_prob = evaluator.get_log_prob(recon_x, mu=x_mu, std=x_std)
#
#                     # recon_abs
#                     _, recall_rec, prec_rec, th_rec, f1_rec = evaluator.pr_auc((recon_x - x).abs(),y)
#                     argmax = np.argmax(f1_rec)
#                     recall_rec, prec_rec, th_rec, f1_rec =\
#                         recall_rec[argmax], prec_rec[argmax], th_rec[argmax], f1_rec[argmax]
#
#                     print('reconstruction loss (abs)')
#                     print('recall :{0:.4f}, precision :{1:.4f}, threshold :{2:.4f},'
#                           ' f1 score :{3:.4f}'.format(recall_rec, prec_rec, th_rec, f1_rec))
#
#                     _, recall_prob, prec_prob, th_prob, f1_prob = evaluator.pr_auc(recon_prob, y)
#                     argmax = np.argmax(f1_prob)
#                     recall_prob, prec_prob, th_prob, f1_prob = \
#                         recall_prob[argmax], prec_prob[argmax], th_prob[argmax], f1_prob[argmax]
#
#                     print('reconstruction probability')
#                     print('recall :{0:.4f}, precision :{1:.4f}, threshold :{2:.4f},'
#                           ' f1 score :{3:.4f}'.format(recall_prob, prec_prob, th_prob, f1_prob))
#                 break
#
# for i, test_loader in enumerate(test_dataloaders):
#     for b, data in enumerate(test_loader):
#         x, y = data['value'].to(device), data['label'].to(device)
#         x, y = Variable(x), Variable(y)
#
#         optimizer.zero_grad()
#
#         x_mu, x_logvar, z_mean_, z_logvar_ = net(x)
#
#         x_std = x_logvar.mul(0.5).exp_()
#         recon_x = sampling(mu=x_mu, sigma=x_std)
#
#         z_std_ = z_logvar_.exp_.sqrt()
#
#         z = sampling(sigma=z_std, size=(z_mean_.size()))
#         z_tilde = z_mean_ + (z_std_.exp() + 1e-8).sqrt() * torch.randn(z_mean_.size()).to(device)
#
#         # mmd loss
#         MMD = mmd(z_tilde, z, z_var=z_var)
#
#         recon_loss = F.mse_loss(recon_x, x, size_average=False).div(batch_size)
#         if clipping is False:
#             eps = torch.randn(z_mean_.size(0), hidden_n).to(device)
#             z_hat = eps * z + (1 - eps) * z_tilde
#             z_output, _, __, ___ = net(z_hat, True)
#             z_d = \
#                 autograd.grad(outputs=z_output, inputs=z_hat, grad_outputs=torch.ones(z_output.size()).to(device))[
#                     0]
#             grad_reg = ((z_d.norm(2) - 1) ** 2).div(batch_size)
#         else:
#             grad_reg = 0
#         loss = recon_loss + MMD + grad_reg
#         recon_prob = evaluator.get_log_prob(recon_x, mu=x_mu, std=x_std)
#
#         # recon_abs
#         _, recall_rec, prec_rec, th_rec, f1_rec = evaluator.pr_auc((recon_x - x).abs(),y)
#         argmax = np.argmax(f1_rec)
#         recall_rec, prec_rec, th_rec, f1_rec =\
#             recall_rec[argmax], prec_rec[argmax], th_rec[argmax], f1_rec[argmax]
#
#         print('reconstruction loss (abs)')
#         print('recall :{0:.4f}, precision :{1:.4f}, threshold :{2:.4f},'
#               ' f1 score :{3:.4f}'.format(recall_rec, prec_rec, th_rec, f1_rec))
#
#         _, recall_prob, prec_prob, th_prob, f1_prob = evaluator.pr_auc(recon_prob, y)
#         argmax = np.argmax(f1_prob)
#         recall_prob, prec_prob, th_prob, f1_prob = \
#             recall_prob[argmax], prec_prob[argmax], th_prob[argmax], f1_prob[argmax]
#
#         print('reconstruction probability')
#         print('recall :{0:.4f}, precision :{1:.4f}, threshold :{2:.4f},'
#               ' f1 score :{3:.4f}'.format(recall_prob, prec_prob, th_prob, f1_prob))



    #     anomaly = []
    #     for idx in y[:]:
    #         if idx[idx == 1].shape[0] > 0:
    #             anomaly.append(1)
    #             cnt += 1
    #         else:
    #             anomaly.append(0)
    #
    #     z_.extend(z.cpu().detach().numpy())
    #     a_.extend(anomaly)
    #     l_.extend([loss.cpu().detach().numpy() / batch_size] * batch_size)
    # result = [z_, a_, l_]
    # result = map(list, zip(*result))
    # result = pd.DataFrame(result, columns=['z', 'anomaly', 'loss'])
    # if plotting:
    #     # result, B, k
    #     shape = (result.shape[0] / batch_size, batch_size, np.shape(result.iloc[0]['z'])[0])
    #     z = torch.as_tensor(result['z']).view(-1, shape[2])
    #
    #     z_mean = torch.mean(z, 0, keepdim=True)  # mean by k_dim
    #     z = z - z_mean.expand_as(z)
    #     U, S, V = torch.svd(z)
    #     # extract 2 component using PCA
    #     z = U[:, :2]  # result,B,2
    #     result['z_x'], result['z_y'] = z[:, 0].numpy(), z[:, 1].numpy()
    #     groups = result.groupby('anomaly')
    #
    #     label = ['normal', 'anomaly']
    #     marker = ['o', 's']
    #     fig, ax = plt.subplots()
    #
    #     for anomaly, group in groups:
    #         ax.plot(group.z_x, group.z_y, marker=marker[anomaly], linestyle='', label=label[anomaly])
    #     ax.legend()
    #     plt.show()
    #
    # print('anomaly count : ', cnt)
    # result = result.sort_values(by=['loss'], ascending=False)
    # # cnt = result[result['anomaly']==1].shape[0](result[0:cnt]['anomaly']==1)
    # subset = result[0:cnt]
    # acc = subset[subset['anomaly'] == 1].shape[0]
    # if tensorboard:
    #     writer.add_scalar('recall/anomaly_21', (acc / cnt), i)
    # print('statistics')
    # print('Total : {0} count : {1} accuracy : {2:.5f}'.format(cnt, acc, acc / cnt))