import torch
from pre_processing import Preprocessing
from models import HaarVAE, VAENet
import torch.optim as optim
from data_ import fast_loader
from utils import get_log_prob, plot_result, get_result, anomaly_is_there
import numpy as np
import matplotlib.pyplot as plt
from plot_ import plot_result_horizon
from tensorboardX import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from dataprocessing import wavelet_transform

# VAE setting
epoch = 6000
q_size = 700
batch_size = 1024
input_size = q_size
hidden_size = 30
output_size = q_size
k = 3
lr = 0.0001
standardized = True
low_frq_remove = False
normalized = False
window_stand = False
batch_version = False

if not batch_version:
    batch_size = -1

# setting device and default data type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# data_loader = fast_loader('yahoo','A3Benchmark')
data_loader = fast_loader('nab','realKnownCause')
# data_loader = fast_loader('kpi')

# data-preprocessing
for xs, ys, title in data_loader:
    preprocessor = Preprocessing(xs, ys, q_size, batch_size, device,
                                 standardization=standardized, remove_low_freq=low_frq_remove, window_standardization=window_stand, scaling=normalized)
    train_x, train_y, test_x, test_y = preprocessor.get_data()
    print('Data are ready')

    train_idx_anomaly, train_idx_normal, test_idx_anomaly, test_idx_normal = preprocessor.get_index()

    anomaly_is_there(test_idx_anomaly)

    # plotting
    window = plt.figure()
    te_l = window.add_subplot(311)
    te_r = window.add_subplot(312)
    te_m = window.add_subplot(313, projection='3d')
    # f1.suptitle(title + '_train')
    window.suptitle(title + '_test')
    writer = SummaryWriter(logdir='wavelet')

    net = HaarVAE(input_size, k, hidden_size, device).to(device)
    # net = VAENet(input_size, k, input_size, hidden_size, device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)
    print('training is started')

    batch_len = len(train_x)

    for e in range(epoch):
        valid = 0
        for idx_b, data in enumerate(train_x):

            optimizer.zero_grad()

            x_mu, x_logvar, z_mu, z_logvar, z = net(data)

            x_std = x_logvar.mul(0.5).exp_()
            x_std = torch.clamp_max(x_std,1)
            z_std = z_logvar.mul(0.5).exp_()
            data = data.view(-1,output_size)
            KLD = -0.5 * (1 + z_logvar - z_mu**2 - z_logvar.exp()).sum(-1)

            recon_prob = (get_log_prob(data, x_mu, x_logvar)) #+ get_log_prob(z, 0,1).sum(-1).unsqueeze(1))

            N_recon = recon_prob[train_idx_anomaly[idx_b]].sum(-1).exp().mean()
            A_recon = recon_prob[train_idx_anomaly[idx_b]].sum(-1).exp().mean()
            N_z = z[train_idx_normal[idx_b]]
            A_z = z[train_idx_anomaly[idx_b]]

            loss = -(recon_prob.sum(-1)).mean() + KLD.mean()

            true, prec, rec, bestf1_z = get_result(-get_log_prob(z, 0, 1).detach().cpu().numpy().sum(-1), train_idx_anomaly[idx_b].detach().cpu().numpy())
            true, prec, rec, bestf1_p = get_result(-recon_prob.sum(-1).detach().cpu().numpy(), train_idx_anomaly[idx_b].detach().cpu().numpy())
            true, prec, rec, bestf1_x = get_result((data - x_mu).abs().sum(-1).detach().cpu().numpy(), train_idx_anomaly[idx_b].detach().cpu().numpy())
            writer.add_scalars('loss', {'train loss': loss}, e*batch_len+idx_b+1)

            # writer.add_scalars('N-A ratio', {'train':  N_recon/ A_recon }, e + 1)
            # writer.add_scalars('KL', {'Normal' : get_log_prob(N_z,0,1).sum(-1).exp().mean(),
            #                           'Anomaly': get_log_prob(A_z,0,1).sum(-1).exp().mean()}, e+1)
            writer.add_scalars('Best f1-score', {'train - z likelihood': bestf1_z},e*batch_len +idx_b+1)
            writer.add_scalars('Best f1-score', {'train - recon probs': bestf1_p}, e*batch_len +idx_b + 1)
            writer.add_scalars('Best f1-score', {'train - recon loss': bestf1_x}, e*batch_len +idx_b + 1)
            valid += loss
            loss.backward()
            optimizer.step()
        print('{1} loss : {0:.4f}'.format(valid, e+1))

        ###################### FOR TEST #######################
        if (e+1) % 50 == 0:
            data, label = test_x, test_y
            optimizer.zero_grad()
            x_mu, x_logvar, z_mu, z_logvar, z = net(data)
            z_std = z_logvar.mul(0.5).exp_()
            x_std = x_logvar.mul(0.5).exp_()
            x_std = torch.clamp_max(x_std, 1)
            data = data.view(-1, q_size)

            KLD = -0.5 * (1 + z_logvar - z_mu ** 2 - z_logvar.exp()).sum(-1)
            recon_prob = (get_log_prob(data, x_mu, x_logvar))  # + get_log_prob(z, 0,1).sum(-1).unsqueeze(1))


            N_recon = recon_prob[test_idx_normal].exp().mean()
            A_recon = recon_prob[test_idx_anomaly].exp().mean()
            N_z = z[test_idx_normal]
            A_z = z[test_idx_anomaly]

            true, prec, rec, bestf1_z = get_result(-get_log_prob(z, 0, 1).detach().cpu().numpy().sum(-1), test_idx_anomaly.detach().cpu().numpy())
            true, prec, rec, bestf1_p = get_result(-recon_prob.sum(-1).detach().cpu().numpy(), test_idx_anomaly.detach().cpu().numpy())
            true, prec, rec, bestf1_x = get_result((data - x_mu).abs().sum(-1).detach().cpu().numpy(), test_idx_anomaly.detach().cpu().numpy())

            # writer.add_scalars('N-A ratio', {'test': N_recon / A_recon},e + 1)
            # writer.add_scalars('KL/test', {'Normal': get_log_prob(N_z, 0, 1).sum(-1).exp().mean(),
            #                           'Anomaly': get_log_prob(A_z, 0, 1).sum(-1).exp().mean()}, e + 1)
            writer.add_scalars('Best f1-score', {'test - z likelihood': bestf1_z}, e * batch_len + 1)
            writer.add_scalars('Best f1-score', {'test - recon probs': bestf1_p}, e * batch_len + 1)
            writer.add_scalars('Best f1-score', {'test - recon loss': bestf1_x}, e * batch_len + 1)

            x_s, y_s = data[:,-1].view(-1).detach().cpu().numpy(), label.view(-1).detach().cpu().numpy()
            recon_probs = recon_prob[:,-1].view(-1).detach().cpu().numpy()
            recons = x_mu[:,-1].contiguous().view(-1).detach().cpu().numpy()
            stds = x_std[:,-1].contiguous().view(-1).detach().cpu().numpy()
            zs= z.detach().cpu().numpy()
            plot_result_horizon(x_s, y_s, recon_probs, recons, stds, te_l, te_r, te_m, zs, sliding=True, q_size=q_size)

    print('the end')
    plt.close()