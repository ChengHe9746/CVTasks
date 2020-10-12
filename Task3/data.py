# -*- coding: utf-8 -*-
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch
from torch.utils.data import DataLoader


def get_data(batch_size, root='/mnist_data', train=True, val=True, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    print('mnist data loader with %d workers...' % (num_workers))
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data = []
    if train:
        train_data = datasets.MNIST(root=root, train=True, download=True,
                                    transform=transformer)
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, **kwargs)
        data.append(train_loader)
    if val:
        test_data = datasets.MNIST(root=root, train=False, download=True,
                                   transform=transformer)
        test_loader = DataLoader(test_data, batch_size=batch_size,
                                 shuffle=True, **kwargs)
        data.append(test_loader)
    data = data[0] if len(data) == 1 else data
    return data










