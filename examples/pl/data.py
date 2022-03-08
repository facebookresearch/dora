# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl


def get_datasets(root: str):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)

    return trainset, testset


class DataModule(pl.LightningDataModule):
    def __init__(self, root: str, batch_size: int = 32):
        super().__init__()
        self.root = root
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        get_datasets(self.root)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: tp.Optional[str] = None):
        self.train, self.test = get_datasets(self.root)

    # return the dataloader for each split
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=10)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=10)
