from scipy import signal
import matplotlib.pyplot as plt
from data_ import KPI, Yahoo
import numpy as np
from model import ConvAE
from WAE import WAE
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch.optim as optim
import torch
from torch.autograd import Variable
from Evaluation import evaluate
from Converge import converge
import time
from torch.optim.lr_scheduler import StepLR
from SpectralResidual import saliency_map



input_size=(513,14)

q_size = 120
k = 10
# output_size=5
in_channels = 513
lr = 0.01
train_dataloaders = Yahoo('train', dir='A3Benchmark', norm=True, q_size=120, batch_size=128, ratio=0.7)
test_dataloaders = Yahoo('test', dir='A3Benchmark', norm=True, q_size=120, batch_size=128, ratio=0.7)
nperseg = 10
noverlap = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for id, (train_dataloader, test_dataloader) in enumerate(zip(train_dataloaders, test_dataloaders)):
    valid = 0
    evaluator = evaluate('yahoo')
    net = ConvAE(k, in_channels).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
    converger = converge()
    file_id = train_dataloaders.files[train_dataloaders.idx]
    cnt = 0
    start_time = time.time()
    break_point = False
    # if id == 0:
    #     print('id 0 is skipped')
    #     continue
    for e in range(500):
        scheduler.step()
        if not break_point:
            valid = 0
            start = 0
            for i, data in enumerate(train_dataloader):
                #
                # print('===========================================',id,' ',i)
                # if i >= 3:
                #     break
                fs = 10e3
                x, y = data['value'], data['label']
                # cnt = len(y[0][y[0]==1])
                # print(cnt)

                input_Sxx = []
                for i in range(x.size(0)):
                    if i == 0:
                        f, t, Sxx = signal.spectrogram(x[i].data.cpu().numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=1024)
                    else:
                        _, __, Sxx = signal.spectrogram(x[i].data.cpu().numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=1024)
                    input_Sxx.append(Sxx)
                x, y = Variable(torch.FloatTensor(input_Sxx)).to(device), Variable(y).to(device)
                optimizer.zero_grad()

                recon_x = net(x)
                y_raw = (recon_x - x).pow(2).sum(1)
                y_raw_p = (recon_x - x).pow(2)
                loss = y_raw.sum()

                valid += loss
                loss.backward()
                optimizer.step()

            print('loss {0:.4f}'.format(valid))
            new_labels = None

            while(start+nperseg < y.size(1)):
                tmp = y[:,start:start+nperseg]
                new_label= tmp.sum(1).view(-1,1)
                if new_labels is None:
                    new_labels = new_label
                else:
                    new_labels = torch.cat((new_labels,new_label), dim=1)
                start = start + nperseg-noverlap
            new_labels = torch.where(new_labels > 0, torch.ones_like(new_labels), new_labels)
            evaluator.get_result(y_raw, new_labels)

            if converger.check_converge(valid):
                evaluator.record(e)
                for i, test_data in enumerate(test_dataloader):
                    x, y = test_data['value'], test_data['label']
                    f, t, Sxx = signal.spectrogram(x[i].data.cpu().numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=1024)
                    Sxx, y = [Sxx], y.unsqueeze(0)
                    x, y = Variable(torch.FloatTensor(Sxx)).to(device), Variable(y).to(device)
                    optimizer.zero_grad()

                    recon_x = net(x)
                    y_raw_p = (recon_x - x).pow(2)
                    start = 0
                    new_labels = None
                    while (start + nperseg < y.size(1)):
                        tmp = y[i, start:start + nperseg]
                        new_label = tmp.sum().view(1, 1)
                        if new_labels is None:
                            new_labels = new_label
                        else:
                            new_labels = torch.cat((new_labels, new_label), dim=1)
                        start = start + nperseg - noverlap

                    cnt += 1
                    ax1 = plt.figure()
                    plt.pcolormesh(t, f, 10*np.log10(x[0].detach().cpu().numpy()))
                    ax2 = plt.figure()
                    plt.pcolormesh(t, f, 10*np.log10(recon_x[0].detach().cpu().numpy()))
                    ax3 = plt.figure()
                    plt.pcolormesh(t, f, 10*np.log10(y_raw_p[0].detach().cpu().numpy()))
                    ax4 = plt.figure()
                    plt.plot(new_labels[0].detach().cpu().numpy())
                    ax5 = plt.figure()
                    plt.plot(saliency_map(x[0]).detach().cpu().numpy())
                    plt.show()
                    break
            if cnt == 10:
                print('end of training for ', file_id)
                break_point = True
                print('training time ', time.time() - start_time)
                # evaluator.get_record(file_id)
                break

















        # plt.plot(Sxx)
        # plt.show()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(t, f, 10*np.log10(Sxx))
        # plt.show()
        # plt.pcolormesh()
        # bx = plt.figure()
        # plt.plot(x[0].data.cpu().numpy())

        # if cnt != 0:
        #     x_ = [i for i,d in enumerate(y[0].cpu().numpy()) if d == 1]
        #     print(x_)
        #     plt.plot(x_, x[0].data.cpu().numpy()[x_],'ro')
    # plt.show()
        # recon_x = net(x)

