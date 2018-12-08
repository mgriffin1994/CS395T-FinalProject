from torchvision import datasets, transforms
import torch
from base import BaseDataLoader
from utils.database import ModelReader

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
    def __init__(self, batch_size):
        super(JointAngleDataset, self).__init__()
        self.batch_size = batch_size
        self.mr = ModelReader(batch_size)
        self.data = np.array([self.mr.prepare_sample(grasp_data)['grasp_grasp_joints'][1:] for  grasp_data in self.mr.getAll()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx] / 2) # Joint angles naturally between -0.5 and 2


class JointAngleDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = JointAngleDataset(batch_size)
        super(JointAngleDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

        
