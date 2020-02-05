import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from data_ import Yahoo
import numpy as np
from Evaluation import evaluate

# from visualization import create_vis_plot
# from visualization import update_vis_plot
#
# from data_loader import Data_loader

batch_size = 1
q_size = 200
# train_dataset = KPI(csv_file='data/KPI/phase2_train.csv',
#                                            root_dir='data/KPI/', q=q_size)
# test_dataset = KPI(csv_file='data/KPI/phase2_test.csv',
#                                            root_dir='data/KPI/', q=q_size)
#
# trainloader = torch.utils.data.DataLoader(train_dataset,
#                                              batch_size=batch_size, shuffle=False,
#                                              num_workers=4)
# testloader = torch.utils.data.DataLoader(test_dataset,
#                                              batch_size=batch_size, shuffle=False,
#                                              num_workers=4)

# print('Data configuration')
# print('Train :', len(train_dataset))
# print('Test :', len(test_dataset))

train_dataloaders = Yahoo('train', dir='A3Benchmark', norm=True, q_size=q_size, batch_size=batch_size, ratio=0.7)
test_dataloaders = Yahoo('test', dir='A3Benchmark', norm=True, q_size=q_size, batch_size=batch_size, ratio=0.7)
data_type ='yahoo'



# RNN
input_size = 1
hidden_size = 10
batch_size_rnn = 1

# VAE setting
input_n = hidden_size
hidden_n = 5
output_size = q_size

# latent dim (set 2 for plotting space)
k = 2
n_test = 4
n_epoch = 1
n_valid = 100
# batch_size = 128
lr = 0.001
wd_l2=0.001
kl_weight = 0.0
tau = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = F.relu(out[:, -1])
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

class VAENet(nn.Module):
    def __init__(self):
        super(VAENet, self).__init__()
        # encoding
        self.fc1 = nn.Linear(input_n, hidden_n)
        # mu & std
        self.fc21 = nn.Linear(hidden_n, k)
        self.fc22 = nn.Linear(hidden_n, k)
        # decoding
        self.fc3 = nn.Linear(k, hidden_n)
        self.fc4 = nn.Linear(hidden_n, output_size)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparametrizing(mu, log_var)
        recon_x = self.decoding(z)

        return recon_x, mu, log_var

    def encoding(self, x):
        x = F.relu(self.fc1(x))
        mu = F.relu(self.fc21(x))
        log_var = F.relu(self.fc22(x))
        return mu, log_var

    def reparametrizing(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_(mean=0, std=1).cuda()
        z = eps.mul(std).add_(mu)
        return z

    def decoding(self, z):
        recon_x = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(recon_x))
        return recon_x

# weight initialization for VAE
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('fc') != -1:
        torch.nn.init.kaiming_normal(m.weight.data)
        torch.nn.init.kaiming_normal(m.bias.data)

# input_size to hidden_size
rnn = GRUNet(input_size, hidden_size).to(device)

net = VAENet().to(device)
net.apply(weight_init)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.BCELoLoss(reduction='sum')

# training
for e in range(n_epoch):
    for idx, trainloader in enumerate(train_dataloaders):
        valid_loss = 0.0
        for b, data in enumerate(trainloader):
            x, y = data['value'], data['label']
            x, y = Variable(x.type(torch.FloatTensor)).to(device), Variable(y.type(torch.DoubleTensor)).to(device)
            x = x.view(-1,q_size,1) # Batch,Time,Depth
            optimizer.zero_grad()
            h = rnn.init_hidden(batch_size).data
            output, h = rnn(x, h)
            # recon loss
            recon_x, mu, log_var = net(output)
            # KL loss
            # = 0.5 * sum(mu^2 + sigma^2 + log(sigma^2) - 1)
            KLD = torch.sum(mu ** 2 + log_var.exp() - (log_var) - 1).mul_(0.5)

            std_ = torch.exp(log_var) ** (1 / 2)
            x = x.view(-1,output_size)

            loss = criterion(recon_x, x) + KLD

            loss.backward()
            optimizer.step()

            valid_loss += loss.item()
            if (b % n_valid) == (n_valid - 1):
                print('Train || {0:2d} epoch {1:4d} batch : loss : {2:.5f}'.format(e, b + 1, valid_loss / (n_valid * batch_size)))
                valid_loss = 0.0
            # update_vis_plot(e * len(trainloader) + b, loss, iter_plot, 'append')
            # update_vis_plot(e * len(trainloader) + b, torch.mean(x), value_plot, 'append')
        evaluator = evaluate('yahoo')
        for test_i, testloader in enumerate(test_dataloaders):
            for i, data in enumerate(testloader):
                x, y, idx = data['value'], data['label'], data['index']
                x = x.view(x.size(0), -1)
                x, y = Variable(x).to(device), Variable(y).to(device)

                optimizer.zero_grad()

                # recon loss
                recon_x, mu, log_var = net(x)

                # KL loss
                # = 0.5 * sum(mu^2 + sigma^2 + log(sigma^2) - 1)
                KLD = torch.sum(mu ** 2 + log_var.exp() - (log_var) - 1).mul_(0.5)

                std_ = torch.exp(log_var) ** (1 / 2)
                recon_loss = criterion(recon_x, x)
                loss = recon_loss  + KLD

                _, recall_rec, prec_rec, th_rec, f1_rec = evaluator.pr_auc(recon_loss, y)
                argmax = np.argmax(f1_rec)
                recall_rec, prec_rec, th_rec, f1_rec = \
                    recall_rec[argmax], prec_rec[argmax], th_rec[argmax], f1_rec[argmax]

                print('reconstruction loss')
                print('recall :{0:.4f}, precision :{1:.4f}, threshold :{2:.4f},'
                      ' f1 score :{3:.4f}'.format(recall_rec, prec_rec, th_rec, f1_rec))
                print('loss {0}'.format(loss.item()))

                break
            break
#
# test_anomalies_count = [1 for i in train_dataset.data['label'].values if i==1]
# true_anomalies = len(test_anomalies_count)
# print('# of true anomalies :', true_anomalies)
#
# # sorting loss
# sorted_index = np.argsort(loss)
#
# print('value    label   loss')
# pre_c = 0 # the number of precision
# corr = 0 # the number of correct precision
# for i in range(sorted_index):
#     if loss[i] < tau:
#         break
#     print('{0}  {1} {2}'.format(x[i], y[i], loss[i]))
#     pre_c += 1
#     if y[i] == 1:
#         corr += 1
# precision = corr/pre_c
# recall = corr/true_anomalies
# print('statistics : precision {0} {1}'.format(precision, recall))
