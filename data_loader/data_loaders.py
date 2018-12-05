from torchvision import datasets, transforms
import torch
from base import BaseDataLoader

import numpy as np

class MnistDataLoader(BaseDataLoader):
    """MNIST data loading demo using BaseDataLoader"""
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class CIFAR10DataLoader(BaseDataLoader):
    """CIFAR10 data loading demo using BaseDataLoader"""
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super(CIFAR10DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
       
class JointAngleDataset(torch.utils.data.Dataset):
    # TODO
    def __init__(self):
        super(JointAngleDataset, self).__init__()
        pass

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return np.array([1] * 12).astype(np.float32), 1


class JointAngleDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = JointAngleDataset()
        super(JointAngleDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
