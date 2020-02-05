import torch
from torch.utils import data
import pandas as pd
import numpy as np
import sys
import os


class Sequential(data.Dataset):
    def __init__(self, csv_file, q_size, norm=True, train='train', ratio=0.7, x='value', y='label', sliding=True):
        self.q = q_size
        self.x = x
        self.y = y
        self.sliding = sliding
        print(csv_file)
        if isinstance(csv_file, str):
            data = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            data = csv_file
        else:
            print('data_/Sequential')
            print('data type is not defined')
            sys.exit()

        if train == 'train':
            self.data = data.iloc[:int(np.floor(len(data) * ratio))]
        else:
            self.data = data.iloc[int(np.floor(len(data) * ratio)):]
        if norm:
            min = self.data[x].min()
            range_ = self.data[x].max() - min
            self.data[x] = (self.data[x] - min) / range_
        self.data_length = len(self.data)

    def __len__(self):
        if self.sliding:
            length = int(len(self.data) - (self.q - 1))
        else:  # cutting
            length = len(self.data) // int(self.q)
        return length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx + self.q > self.data_length:
            idx_fin = -1
        else:
            idx_fin = idx + self.q
        value = self.data.iloc[idx:idx_fin][self.x].values
        label = self.data.iloc[idx:idx_fin][self.y].values

        sample = {'value': torch.from_numpy(value).type(torch.FloatTensor),
                  'label': torch.from_numpy(label).type(torch.IntTensor)}

        return sample


class KPI():
    def __init__(self, type, norm, q_size, batch_size, ratio, shuffle=True, num_workers=2, sliding=True):
        self.type = type
        self.norm = norm
        self.q_size = q_size
        self.ratio = ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.size = {'train': 29, 'test': 10}[type]

        root = {'train': os.path.join('data', 'KPI', 'phase2_train.csv'),
                'test': os.path.join('data', 'KPI', 'phase2_ground_truth.csv')}

        self.data = pd.read_csv(root[self.type]).rename(columns={'KPI ID': 'ID'})
        self.ids, self.counts = np.unique(self.data['ID'], return_counts=True)
        self.sliding = sliding

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.size:
            raise StopIteration
        print(self.ids[self.idx])
        dataset = Sequential(self.data[self.data.ID == self.ids[self.idx]], self.q_size,
                             norm=self.norm, train=self.type, ratio=self.ratio, sliding=self.sliding)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size, shuffle=self.shuffle,
                                                 num_workers=self.num_workers)
        self.idx += 1
        return dataloader


class NAB(data.Dataset):
    def __init__(self, type, dir, norm, q_size, batch_size, ratio, shuffle=True, num_workers=2, sliding=True):
        self.check_dir(dir)
        self.type = type
        self.norm = norm
        self.q_size = q_size
        self.ratio = ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.size = {'artificialWithAnomaly': 6,
                     'realAdExchange': 6,
                     'realAWSCloudwatch': 17,
                     'realKnownCause': 7,
                     'realTraffic': 7,
                     'realTweets': 10}[dir]
        self.root = os.path.join('data', 'NAB', 'labeled', dir)
        self.files = os.listdir(self.root)
        self.idx = 0
        self.sliding = sliding

    def check_dir(self, dir):
        if dir not in ['artificialWithAnomaly', 'realAdExchange', 'realAWSCloudwatch', 'realKnownCause', 'realTraffic',
                       'realTweets']:
            print('data_/NAB')
            print('there is no', dir)
            sys.exit()

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.size:
            raise StopIteration
        file_root = os.path.join(self.root, self.files[self.idx])

        dataset = Sequential(file_root, self.q_size, norm=self.norm, train=self.type, ratio=self.ratio,
                             sliding=self.sliding)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size, shuffle=self.shuffle,
                                                 num_workers=self.num_workers)
        self.idx += 1
        return dataloader


class Yahoo(data.Dataset):
    def __init__(self, type, dir, norm, q_size, batch_size, ratio, shuffle=True, num_workers=2, sliding=True):
        self.check_dir(dir)
        self.type = type
        self.norm = norm
        self.q_size = q_size
        self.ratio = ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.size = 67

        self.root = os.path.join('data', 'ydata-labeled-time-series-anomalies-v1_0', dir)
        self.files = [f for f in os.listdir(self.root)
                      if f[-4:] == '.csv' and f[-7:-4] != 'all']
        self.label = {'A1Benchmark': 'is_anomaly',
                      'A2Benchmark': 'is_anomaly',
                      'A3Benchmark': 'anomaly',
                      'A4Benchmark': 'anomaly'}[dir]
        self.sliding = sliding

    def check_dir(self, dir):
        if dir not in ['A1Benchmark', 'A2Benchmark', 'A3Benchmark', 'A4Benchmark']:
            print('data_/Yahoo')
            print('there is no', dir)
            sys.exit()

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.size:
            raise StopIteration
        file_root = os.path.join(self.root, self.files[self.idx])
        print(self.files[self.idx])
        dataset = Sequential(file_root, self.q_size, norm=self.norm, train=self.type, ratio=self.ratio, x='value',
                             y=self.label, sliding=self.sliding)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size, shuffle=self.shuffle,
                                                 num_workers=self.num_workers)
        self.idx += 1
        return dataloader

#
#
# class NAB(data.Dataset):
#   def __init__(self, csv_file, q_size):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             q (int): the window size.
#         """
#         self.data = pd.read_csv(csv_file)
#         self.q = q_size
#
#   def __len__(self):
#       return int(np.floor(self.data.shape[0]/self.q))
#
#   def __getitem__(self, idx):
#       if torch.is_tensor(idx):
#           idx = idx.tolist()
#       # timestamp = self.data.iloc[idx*self.q: (idx+1)*self.q]['timestamp'].values
#       value = self.data.iloc[idx*self.q: (idx+1)*self.q]['value'].values
#       label = self.data.iloc[idx*self.q: (idx+1)*self.q]['label'].values
#       index = idx*self.q
#       sample = {'value': value, 'label': label, 'index': index}
#
#       return sample
#
#   def get_label(self):
#       return {'normal': 0, 'abnormal': 1}

# class KPI():
#     def __init__(self, type, norm, q_size):
#         """
#         Args:
#             type (string): train or test.
#             q (int): the window size.
#         """
#         if type in ['train', 'Train']:
#             root = os.path.join('data', 'KPI','phase2_train.csv')
#         elif type in ['test', 'Test']:
#             root = os.path.join('data', 'KPI', 'phase2_ground_truth.csv')
#
#         self.data = pd.read_csv(root).rename(columns={'KPI ID':'ID'})
#         self.idx = []
#         self.id, counts = np.unique(self.data['ID'], return_counts=True)
#         for i in self.id:
#             self.idx.extend(self.data[self.data.ID==i].index[:-(q_size-1)])
#         self.idx = list(self.idx)
#         self.q = q_size
#
#         if norm:
#             for i in self.id:
#                 tmp = self.data.loc[self.data.ID==i,'value']
#                 min, range = tmp.min(), tmp.max() - tmp.min()
#                 self.data.loc[self.data.ID==i,'value'] = (self.data.loc[self.data.ID==i,'value'] - min) / range
#
#     def __len__(self):
#         """
#         :return: randomly choosen id
#         """
#         return len(self.idx)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#           idx = idx.tolist()
#         value = self.data[self.idx[idx]:self.idx[idx]+self.q]['value'].values
#         label = self.data[self.idx[idx]:self.idx[idx]+self.q]['label'].values
#         # index = idx*self.q
#         sample = {'value': torch.from_numpy(value).type(torch.FloatTensor),
#                   'label': torch.from_numpy(label).type(torch.IntTensor)}
#
#         return sample
#
#     def get_label(self):
#         return {'normal': 0, 'abnormal': 1}
#     def __init__(self, dir, type,ratio,norm, q_size):
#         """
#         Args:
#         file (string): type of NAB (eg.,realKnownCause)
#         q (int): the window size.
#         """
#
#
#         self.data = None
#
#             if norm:
#                 min = tmp['value'].min()
#                 range = tmp['value'].max() - min
#                 tmp['value'] = (tmp['value'] - min)/ range
#             if type == 'train':
#                 if i == 0:
#                     self.data = tmp[:int(np.floor(ratio*len(tmp))) -(self.q-1)]
#                 else:
#                     self.data.append(tmp[:int(np.floor(ratio*len(tmp))) -(self.q-1)],ignore_index=True)
#             else:   #test
#                 if i == 0:
#                     self.data = tmp[int(np.floor(ratio * len(tmp))), -(self.q-1)]
#                 else:
#                     self.data.append(tmp[int(np.floor(ratio*len(tmp))), -(self.q-1)],ignore_index=True)
#
#
#         self.q = q_size
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         obj = self.data[idx]
#         s = np.random.randint(len(obj) - self.q + 1)
#         value = self.data[idx].iloc[s:s+self.q]['value'].values
#         label = self.data[idx].iloc[s:s+self.q]['label'].values
#         sample = {'value': torch.from_numpy(value).type(torch.FloatTensor),
#                   'label': torch.from_numpy(label).type(torch.IntTensor)}
#
#         return sample
#
#     def get_label(self):
#         return {'normal': 0, 'abnormal': 1}
#
#
#
#     def __init__(self, dir, type, ratio, norm, q_size):
#         """
#         Args:
#           csv_file (DataFrame): splited dataset.
#           q (int): the window size.
#         """
#         if dir not in ['A1Benchmark','A1Benchmark','A1Benchmark','A1Benchmark']:
#             print('there is no ',dir)
#             sys.exit()
#         root = os.path.join('data','ydata-labeled-time-series-anomalies-v1_0',dir)
#         self.data = []
#         for i, file in enumerate(os.listdir(root)):
#             if file[-4:] == '.csv':
#                 tmp = pd.read_csv(os.path.join(root, file))
#                 i -= 1
#                 if type == 'train':
#                     self.data.append(tmp[:int(np.floor(len(tmp)*ratio))])
#                 else:
#                     self.data.append(tmp[int(np.floor(len(tmp) * ratio)):])
#                 if norm:
#                     min = self.data[i]['value'].min()
#                     range = self.data[i]['value'].max() - min
#                     self.data[i]['value'] = ((self.data[i]['value'] - min)/range).astype(float)
#                     # list(min_max_scaler.fit_transform(self.data[i][['value']].values.astype(float)).reshape(-1))
#         self.q = q_size
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#           idx = idx.tolist()
#         obj = self.data[idx].shape[0]
#         s = np.random.randint(obj - self.q + 1)
#         value = self.data[idx].loc[s:s + self.q,'value'].values
#         label = self.data[idx].loc[s:s + self.q,'is_anomaly'].values
#
#         sample = {'value': torch.from_numpy(value).type(torch.FloatTensor),
#                   'label': torch.from_numpy(label).type(torch.IntTensor)}
#
#         return sample
#
#     def get_label(self):
#         return {'normal': 0, 'abnormal': 1}

