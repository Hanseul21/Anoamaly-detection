import numpy as np
import torch
from data_ import Yahoo
import matplotlib.pyplot as plt

def saliency_map(x):
    if type(x).__module__ == torch.__name__:
        q = x.size(-1)
        input = x.numpy()
    elif type(x).__module__ == np.__name__:
        q = np.shape(x)[-1]
        input = x
    phase = np.fft.fft(input)
    amp = torch.as_tensor(phase.real ** 2 + phase.imag ** 2).sqrt()
    pha = np.arctan(phase.imag/phase.real)
    I = torch.ones(q, q).type(torch.DoubleTensor)/(q*q)
    R = (amp.log() - torch.matmul(amp.log(),I))
    phase.real = R.numpy()
    phase.imag = pha
    # plt.figure()
    # plt.plot(R.numpy())
    # plt.xlabel('Residual')
    # plt.figure()
    # plt.plot(torch.matmul(amp.log(),I).numpy())
    # plt.xlabel('average component')
    # plt.show()


    # phase.real = (torch.as_tensor(phase.real) * R / amp).numpy()
    # phase.imag = (torch.as_tensor(phase.imag) * R / amp).numpy()
    saliency = np.fft.ifft(np.exp(phase))
    # saliency = torch.as_tensor(saliency.real ** 2 + saliency.imag ** 2).sqrt()

    return np.abs(saliency)

test_dataloaders = Yahoo('train','A3Benchmark',True,120,256,0.7)

for i, train_dataloader in enumerate(test_dataloaders):
    if i == 0:
        continue
    for data in train_dataloader:
        x, y = data['value'], data['label']
        # x = torch.as_tensor(x)

        salien = saliency_map(x[0])

        fig = plt.figure()
        fig.add_subplot(2,2,1)
        plt.plot(x.numpy()[0])
        plt.xlabel('raw data')
        fig.add_subplot(2, 2, 2)
        plt.plot(salien)
        plt.xlabel('reconstructed data')
        fig.add_subplot(2,2,3)
        plt.plot(y[0].numpy())
        plt.xlabel('label')
        fig.add_subplot(2,2,4)
        plt.plot(np.abs(x[0].numpy()-salien))
        plt.xlabel('reconstruction loss')
        plt.show()