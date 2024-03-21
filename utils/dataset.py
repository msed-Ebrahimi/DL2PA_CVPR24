import os
import numpy as np
import random
from   six.moves import cPickle as pickle
import torch
import torch.optim as optim
import torch.utils.data as data
from   torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import time
import logging
from pathlib import Path


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

def load_imagenet200(basedir, batch_size, kwargs):
    # Correct basedir.
    basedir += "imagenet200/"

    # Normalization.
    mrgb = [0.485, 0.456, 0.406]
    srgb = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Train loader.
    train_data = datasets.ImageFolder(basedir + "train/",
                                      transform=transforms.Compose([transforms.RandomCrop(64, 4), \
                                                                    transforms.RandomHorizontalFlip(), \
                                                                    transforms.ToTensor(), \
                                                                    normalize]))
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size, shuffle=True, **kwargs)

    # Test loader.
    test_data = datasets.ImageFolder(basedir + "test/",
                                     transform=transforms.Compose([transforms.ToTensor(), \
                                                                   normalize]))
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=batch_size, shuffle=False, **kwargs)

    return trainloader, testloader

def load_imagenet200(basedir, batch_size, kwargs):
    # Correct basedir.
    basedir += "imagenet200/"

    # Normalization.
    mrgb = [0.485, 0.456, 0.406]
    srgb = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mrgb, std=srgb)

    # Train loader.
    train_data = datasets.ImageFolder(basedir + "train/",
                                      transform=transforms.Compose([transforms.RandomCrop(64, 4),
                                                                    transforms.RandomHorizontalFlip(),
                                                                    transforms.ToTensor(),
                                                                    normalize]))
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size, shuffle=True, **kwargs)

    # Test loader.
    test_data = datasets.ImageFolder(basedir + "test/",
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   normalize]))
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=batch_size, shuffle=False, **kwargs)

    return trainloader, testloader

class IMBALANCECIFAR10(datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100(datasets.CIFAR100):
    cls_num = 100

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR100, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCESTL10(datasets.stl10.STL10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCESTL10, self).__init__(root, split, None, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCESVHN(datasets.SVHN):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, split="train",
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCESVHN, self).__init__(root, split, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class LT_Dataset(Dataset):
    num_classes = 1000

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        self.class_map = [0 for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.class_map[sorted_classes[i]] = i

        self.targets = np.array(self.class_map)[self.targets].tolist()

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.targets)):
            j = self.targets[i]
            self.class_data[j].append(i)

        self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


class LT_Dataset_Eval(Dataset):
    num_classes = 1000

    def __init__(self, root, txt, class_map, transform=None):
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.class_map = class_map
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.targets = np.array(self.class_map)[self.targets].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def CIFAR10_LT(distributed, root='./data/cifar10', imb_type='exp',imb_factor=0.01, batch_size=128, num_works=40):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if imb_factor < 1:
        print('Long Tail Setting ...')
        train_dataset = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0,
                                         train=True, download=True, transform=train_transform)
        cls_num_list = train_dataset.get_cls_num_list()
    else:
        assert (imb_factor == 1)
        print('Normal Case Setting ...')
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True,
                                         transform=train_transform)
    eval_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    train_instance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True, sampler=dist_sampler)

    balance_sampler = ClassAwareSampler(train_dataset)
    train_balance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True, sampler=balance_sampler)

    eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True)

    return train_instance, eval

def CIFAR100_LT(distributed, root='./data/cifar100', imb_type='exp', imb_factor=0.01, batch_size=128, num_works=40):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = IMBALANCECIFAR100(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0,
                                      train=True, download=True, transform=train_transform)
    eval_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)

    cls_num_list = train_dataset.get_cls_num_list()

    dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    train_instance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True, sampler=dist_sampler)

    balance_sampler = ClassAwareSampler(train_dataset)
    train_balance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True, sampler=balance_sampler)

    eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True)

    return train_instance, eval

def ImageNet_LT(distributed, root="", batch_size=60, num_works=40):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_txt = "./datasets/data_txt/ImageNet_LT_train.txt"
    eval_txt = "./datasets/data_txt/ImageNet_LT_test.txt"

    train_dataset = LT_Dataset(root, train_txt, transform=transform_train)
    eval_dataset = LT_Dataset_Eval(root, eval_txt, transform=transform_test, class_map=train_dataset.class_map)

    cls_num_list = train_dataset.cls_num_list

    dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    train_instance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

    balance_sampler = ClassAwareSampler(train_dataset)
    train_balance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True, sampler=balance_sampler)

    eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True)

    return train_instance, eval

def STL10_LT( distributed, root='./data/cifar10', imb_type='exp',
                 imb_factor=0.01, batch_size=128, num_works=40):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    if imb_factor < 1:
        print('Long Tail Setting ...')
        train_dataset = IMBALANCESTL10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0,
                                       split="train", download=True, transform=transform)
        cls_num_list = train_dataset.get_cls_num_list()
    else:
        assert (imb_factor == 1)
        print('Normal Case Setting ...')
        train_dataset = datasets.stl10.STL10(root=root, split="train", download=True, folds=None,
                                             transform=transform)
    eval_dataset = datasets.stl10.STL10(root=root, split="test", download=True, folds=None,
                                        transform=transform)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    train_instance = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=num_works, pin_memory=True, sampler=dist_sampler)

    eval = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_works, pin_memory=True)

    return train_instance, eval
def SVHN_LT(distributed, root='./data/cifar10', imb_type='exp', imb_factor=0.01, batch_size=128, num_works=40):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if imb_factor < 1:
        print('Long Tail Setting ...')
        train_dataset_1 = IMBALANCESVHN(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0,
                                        split="train", download=True, transform=transform)
        train_dataset_2 = IMBALANCESVHN(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0,
                                        split="extra", download=True, transform=transform)

        cls_num_list = train_dataset_2.get_cls_num_list()
    else:
        assert (imb_factor == 1)
        print('Normal Case Setting ...')
        train_dataset_1 = datasets.SVHN(root=root, split="train", download=True, transform=transform)
        train_dataset_2 = datasets.SVHN(root=root, split="extra", download=True, transform=transform)
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
    eval_dataset = datasets.SVHN(root=root, split="test", download=True, transform=transform)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
    train_instance = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=num_works, pin_memory=True, sampler=dist_sampler)

    eval = torch.utils.data.DataLoader(eval_dataset,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=num_works, pin_memory=True)

    return train_instance, eval

class EffectNumSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx]

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

class RandomCycleIter:

    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]

def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1

class ClassAwareSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, num_samples_cls=4, ):
        # pdb.set_trace()
        num_classes = len(np.unique(data_source.targets))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.targets):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples

def get_sampler():
    return ClassAwareSampler


