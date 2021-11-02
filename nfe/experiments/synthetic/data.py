from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nfe.experiments.synthetic.generate import generate, DATA_DIR


def list_datasets():
    check = lambda x: x.is_file() and x.suffix == '.npz'
    file_list = [x.stem for x in DATA_DIR.iterdir() if check(x)]
    file_list += [x.stem for x in (DATA_DIR).iterdir() if check(x)]
    return sorted(file_list)

def load_dataset(name):
    generate()
    if not name.endswith('.npz'):
        name += '.npz'
    loader = dict(np.load(DATA_DIR / name, allow_pickle=True))
    return TimeSeriesDataset(loader['init'][:,None], loader['time'][...,None], loader['seq'])

def get_data_loaders(name, batch_size):
    # Returns input_dim, n_classes=None, 3*torch.utils.data.DataLoader
    trainset, valset, testset = load_dataset(name).split_train_val_test()
    dl_train = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(valset, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainset.dim, None, dl_train, dl_val, dl_test

def get_single_loader(name, batch_size):
    dataset = load_dataset(name)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dl

class TimeSeriesDataset(Dataset):
    def __init__(self, initial, times, values):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if isinstance(initial, torch.Tensor):
            self.initial = initial
            self.times = times
            self.values = values
        else:
            self.initial = torch.Tensor(initial).to(device)
            self.times = torch.Tensor(times).to(device)
            self.values = torch.Tensor(values).to(device)

    def split_train_val_test(self, train_size=0.6, val_size=0.2):
        ind1 = int(len(self.initial) * train_size)
        ind2 = ind1 + int(len(self.initial) * val_size)

        trainset = TimeSeriesDataset(self.initial[:ind1], self.times[:ind1], self.values[:ind1])
        valset = TimeSeriesDataset(self.initial[ind1:ind2], self.times[ind1:ind2], self.values[ind1:ind2])
        testset = TimeSeriesDataset(self.initial[ind2:], self.times[ind2:], self.values[ind2:])

        return trainset, valset, testset

    @property
    def dim(self):
        return self.values[0].shape[-1]

    def __getitem__(self, key):
        return self.initial[key], self.times[key], self.values[key]

    def __len__(self):
        return len(self.initial)

    def __repr__(self):
        return f'TimeSeriesDataset({self.__len__()})'
