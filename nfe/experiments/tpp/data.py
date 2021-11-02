import numpy as np
import torch

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from nfe.experiments.tpp.generate import generate


DATA_DIR = Path(__file__).parents[1] / 'data/tpp'
DATA_DIR.mkdir(parents=True, exist_ok=True)
ALT_DIR = Path('/opt/ml/input/data/training')


def load_dataset(name, device='cpu'):
    generate()
    if not name.endswith('.npz'):
        name += '.npz'
    filename = DATA_DIR / name
    if not filename.exists():
        filename = ALT_DIR / name
    loader = dict(np.load(filename, allow_pickle=True))
    times = loader['data']
    mask = loader.get('mask', np.ones_like(times))
    marks = loader.get('marks', np.zeros_like(times))
    return TPPDataset(times[...,None], mask[...,None], marks[...,None], device)


def get_data_loaders(name, batch_size, device):
    # Returns input_dim, n_classes, 3*torch.utils.data.DataLoader
    dataset = load_dataset(name, device)
    trainset, valset, testset = dataset.split_train_val_test()
    dl_train = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(valset, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(testset, batch_size=batch_size, shuffle=False)
    n_classes = max(trainset.n_classes, valset.n_classes, testset.n_classes)
    return dataset.dim, dataset.n_classes, dl_train, dl_val, dl_test

class TPPDataset(Dataset):
    def __init__(self, times, mask, marks, device):
        self.device = device
        self.times = torch.tensor(times, dtype=torch.float32, device=device)
        self.mask = torch.tensor(mask, dtype=torch.float32, device=device)
        self.marks = torch.tensor(marks, dtype=torch.long, device=device)

    def split_train_val_test(self, train_size=0.6, val_size=0.2):
        ind1 = int(len(self.times) * train_size)
        ind2 = ind1 + int(len(self.times) * val_size)

        trainset = TPPDataset(self.times[:ind1], self.mask[:ind1], self.marks[:ind1], self.device)
        valset = TPPDataset(self.times[ind1:ind2], self.mask[ind1:ind2], self.marks[ind1:ind2], self.device)
        testset = TPPDataset(self.times[ind2:], self.mask[ind2:], self.marks[ind2:], self.device)

        return trainset, valset, testset

    @property
    def dim(self):
        return self.times[0].shape[-1]

    @property
    def n_classes(self):
        return torch.max(self.marks).int().item() + 1

    def __getitem__(self, key):
        return self.times[key], self.marks[key], self.mask[key]

    def __len__(self):
        return len(self.times)

    def __repr__(self):
        return f'TPPDataset({self.__len__()})'
