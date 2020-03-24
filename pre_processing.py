import numpy as np
import torch

class Preprocessing():
    def __init__(self, raw_x, raw_y, window_size, batch_size, device, train_ratio=0.5, step=1,
                 standardization=True, remove_low_freq=False, window_standardization=False, scaling=False, test_whole=True):
        self.q_size = window_size
        train_x, train_y, test_x, test_y = self.cut_window(raw_x, raw_y, train_ratio, step, test_whole)

        if standardization:
            train_x, test_x = self.standardize(train_x, test_x)
        if remove_low_freq:
            train_x, test_x = self.remove_row_frequency(train_x, test_x)
        if window_standardization:
            train_x, test_x = self.standardize_window(train_x, test_x)
        if scaling:
            train_x, test_x = self.scaling(train_x, test_x)

        train_x, train_y = self.make_batch(train_x, train_y, batch_size)

        self.train_x, self.train_y, self.test_x, self.test_y = train_x, train_y, test_x, test_y

        self.as_torch(device=device)


    def cut_window(self, x, y, train_ratio, step, test_whole):
        '''

        :param x: raw data << np.array >>
        :param y: raw label << np.array >>
        :param train_ratio: ratio of train-data (e.g. train-data / raw data) << float >>
        :param step: step of sliding window << int >>
        :param test_whole: True, when you want to validate whole data (includes trained data)
                            False, validate only newly-come data << Boolean >>
        :return: train-data, train-label, test-data, test-label << np.array >
        '''

        train_len = int(x.__len__() * train_ratio) - self.q_size + 1
        if test_whole:
            test_start = 0
        else:
            test_start = train_len + self.q_size
        test_end = int(x.__len__()) - self.q_size + 1

        train_x = np.array([x[idx:idx + self.q_size] for idx in range(0, train_len, step)], dtype=np.float32)
        train_y = np.array([y[idx:idx + self.q_size] for idx in range(0, train_len, step)], dtype=np.float32)
        test_x = np.array([x[idx:idx + self.q_size] for idx in range(test_start, test_end, step)],dtype=np.float32)
        test_y = np.array([y[idx:idx + self.q_size] for idx in range(test_start, test_end, step)],dtype=np.float32)

        print('train len : ', train_x.__len__())
        print('test len : ', test_x.__len__())

        return train_x, train_y, test_x, test_y

    def standardize(self, train_x, test_x):
        mean_ = train_x.mean()
        std_ = train_x.std()
        # standardizing
        train_x = (train_x - mean_) / std_
        test_x = (test_x - mean_) / std_

        print('Standardization is operated')

        return train_x, test_x

    def remove_row_frequency(self, train_x, test_x):
        # removing low frequency
        for i, window in enumerate(train_x):
            tmp = np.fft.fft(window)
            tmp[0] = 0
            train_x[i] = np.fft.ifft(tmp)
        for i, window in enumerate(test_x):
            tmp = np.fft.fft(window)
            tmp[0] = 0
            test_x[i] = np.fft.ifft(tmp)

        print('Fuerier transform is operated')
        return train_x, test_x

    def scaling(self, train_x, test_x):
        min_ = train_x.min()
        range_ = train_x.max() - min_
        train_x = (train_x - min_) / range_
        test_x = (test_x - min_) / range_

        print('Normalization is operated')
        return train_x, test_x

    def standardize_window(self, train_x, test_x):
        train_x = np.array([(data - np.mean(data)) / np.std(data) for data in train_x])
        test_x = np.array([(data - np.mean(data)) / np.std(data) for data in test_x])

        print('Window standardization is operated')
        return train_x, test_x

    def make_batch(self, train_x, train_y, batch_size):
        # full-batch
        if batch_size == -1:
            train_x = np.expand_dims(train_x, 0)
            train_y = np.expand_dims(train_y, 0)
        else:
            train_idxs = [idx for idx in range(0, train_x.__len__(), batch_size)]
            train_x = np.array([train_x[train_idxs[i]:train_idxs[i+1]] for i in range(0, len(train_idxs)-1)])
            train_y = np.array([train_y[train_idxs[i]:train_idxs[i + 1]] for i in range(0, len(train_idxs) - 1)])

        return train_x, train_y

    def as_torch(self, device):

        self.train_x = torch.from_numpy(self.train_x).to(device)
        self.train_y = torch.from_numpy(self.train_y).to(device)
        self.test_x = torch.from_numpy(self.test_x).to(device)
        self.test_y = torch.from_numpy(self.test_y).to(device)

    def get_data(self):
        '''

        :param as_torch: True, return data as Torch.tensor type
        :return: data and labels
        '''
        print('size of data')
        print(self.train_x.size())
        print(self.test_x.size())

        return self.train_x, self.train_y, self.test_x, self.test_y

    def get_index(self):
        # dtype and device for index
        train_idx_anomaly = torch.where(self.train_y.sum(-1)> 0, torch.ones_like(self.train_y.sum(-1)), torch.zeros_like(self.train_y.sum(-1))).type(torch.ByteTensor)
        train_idx_normal = torch.where(self.train_y.sum(-1) == 0, torch.ones_like(self.train_y.sum(-1)), torch.zeros_like(self.train_y.sum(-1))).type(torch.ByteTensor)
        test_idx_anomaly = torch.where(self.test_y.sum(-1) > 0, torch.ones_like(self.test_y.sum(-1)), torch.zeros_like(self.test_y.sum(-1))).type(torch.ByteTensor)
        test_idx_normal = torch.where(self.test_y.sum(-1) == 0, torch.ones_like(self.test_y.sum(-1)), torch.zeros_like(self.test_y.sum(-1))).type(torch.ByteTensor)

        print('Labels are ready')

        return train_idx_anomaly, train_idx_normal, test_idx_anomaly, test_idx_normal





