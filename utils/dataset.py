# Most notable helpers:
# - Load datasets.
# - Subsample a dataset.
# - Counter the number of parameters in a model.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
#

import os
import numpy as np
from   six.moves import cPickle as pickle
import torch
import torch.optim as optim
import torch.utils.data as data
from   torchvision import datasets, transforms
import os
import time
import logging
from pathlib import Path

#
# Load the CIFAR 100 dataset.
#
def load_cifar100(basedir, batch_size, kwargs):
    # Input channels normalization.
    mrgb = [0.507, 0.487, 0.441]
    srgb = [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Load train data.
    trainloader = torch.utils.data.DataLoader(datasets.CIFAR100(root=basedir + 'cifar100/', train=True,
                                                                transform=transforms.Compose([
                                                                    transforms.RandomCrop(32, 4),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    normalize,
                                                                ]), download=True),
                                              batch_size=batch_size, shuffle=True, **kwargs)
    # Labels to torch.
    trainloader.dataset.train_labels = torch.from_numpy(np.array(trainloader.dataset.targets))

    # Load test data.
    testloader = torch.utils.data.DataLoader(datasets.CIFAR100(root=basedir + 'cifar100/', train=False,
                                                               transform=transforms.Compose([transforms.ToTensor(),
                                                                                             normalize,
                                                                                             ])),
                                             batch_size=batch_size, shuffle=True, **kwargs)
    # Labels to torch.
    testloader.dataset.test_labels = torch.from_numpy(np.array(testloader.dataset.targets))

    return trainloader, testloader


