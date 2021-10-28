"""
This file contains the definition of different heterogeneous datasets used for training
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
import torch
import numpy as np

from .base_dataset import BaseDataset
from .surreal_dataset import SurrealDataset
import math


def create_dataset(dataset, options, **kwargs):
    len2d_eft=[1000, 14810, 9428, 28344]

    dataset_setting = {
        'all': (['h36m-train', 'lsp-orig', 'coco', 'mpii', 'up-3d'],
                [.3, .1, .2, .2, .2]),
        'itw': (['lsp-orig', 'coco', 'mpii', 'up-3d'],
                [.1, .3, .3, .3]),
        'h36m': (['h36m-train'], [1.0]),
        'up-3d': (['up-3d'], [1.0]),
        'mpii': (['mpii'], [1.0]),
        'mesh': (['h36m-train', 'up-3d'],
                 [.7, .3]),
        'spin': (['h36m-train', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp'],
                 [.3, .1, .15, .15, .2, .1]),
        'eft': (['h36m-train', 'lsp-orig', 'mpii-eft', 'lspet-eft', 'coco-eft', 'mpi-inf-3dhp'],
                 [.3, .1, .15, .15, .2, .1]),
        'eft-all': (['h36m-train', 'lsp-orig', 'mpii-eft', 'lspet-eft', 'coco-eft-all', 'mpi-inf-3dhp', 'up-3d'],
                [.3, .1, .1, .1, .2, .1,  .1]),
        'dsr': (['h36m-train', 'coco-eft', 'mpi-inf-3dhp', '3dpw-train'],
                 [.3, .4, .1, .2]),
        'mix1':(['h36m-train', 'mpi-inf-3dhp', '3dpw-train', 'lsp-orig', 'mpii-eft', 'lspet-eft', 'coco-eft'],
                 [.3, .1, .2] + [0.4* l / sum(len2d_eft) for l in len2d_eft]),
        'mix2': (['h36m-train', 'mpi-inf-3dhp', 'lsp-orig', 'mpii-eft', 'lspet-eft', 'coco-eft'],
                 [.3, .1] + [0.6 * l / sum(len2d_eft) for l in len2d_eft]),
    }

    if dataset in dataset_setting:
        datasets, partition = dataset_setting[dataset]
        return MeshMixDataset(datasets, partition, options, **kwargs)
    else:
        return BaseDataset(dataset, options, **kwargs)


def create_val_dataset(dataset, options):
    # Create dataloader for the dataset
    dataset = BaseDataset(options, dataset,
                          use_augmentation=False,
                          is_train=False, use_IUV=False)
    return dataset


import os
class MeshMixDataset(torch.utils.data.Dataset):

    def __init__(self, datasets, partition, options, **kwargs):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        assert math.isclose(sum(partition), 1, abs_tol=1e-2)
        self.partition = np.array(partition).cumsum()

        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in datasets]
        self.length = max(len(ds) for ds in self.datasets)
        self.dataset_infos = []
        begin_dix = 0
        for (ds_name, ds) in zip(datasets, self.datasets):
            self.dataset_infos.append({
                'ds_name': ds_name,
                'begin_idx': begin_dix,
                'len': len(ds),
            })
            ds.begin_idx = begin_dix
            begin_dix += len(ds)

    def __len__(self):
        """Get the size of the dataset."""
        return self.length

    def __getitem__(self, idx):
        """Given index, sample the data from multiple datasets with the given
        proportion."""
        p = np.random.rand()
        for i in range(len(self.datasets)):
            if p <= self.partition[i]:
                index_new = (idx + np.random.rand()) * len(self.datasets[i]) / self.length
                index_new = int(np.round(index_new)) % (len(self.datasets[i]))
                item = self.datasets[i][index_new]
                opt_idx = self.datasets[i].begin_idx + index_new
                item['opt_idx'] = opt_idx
                return item
        return None
