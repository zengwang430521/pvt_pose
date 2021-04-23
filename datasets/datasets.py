"""
This file contains the definition of different heterogeneous datasets used for training
Codes are adapted from https://github.com/nkolot/GraphCMR
"""
import torch
import numpy as np

from .base_dataset import BaseDataset
from .surreal_dataset import SurrealDataset


def create_dataset(dataset, options, **kwargs):
    dataset_setting = {
        'all': (['h36m_train', 'lsp_orig', 'coco', 'mpii', 'up-3d'],
                [.3, .1, .2, .2, .2]),
        'itw': (['lsp_orig', 'coco', 'mpii', 'up-3d'],
                [.1, .3, .3, .3]),
        'h36m': (['h36m-train'], [1.0]),
        'up-3d': (['up-3d'], [1.0]),
        'mesh': (['h36m_train', 'up-3d'],
                 [.7, .3]),
        'spin': (['h36m-train', 'lsp-orig', 'mpii', 'lspet', 'coco', 'mpi-inf-3dhp'],
                 [.3, .1, .15, .15, .2, .1])
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


class MeshMixDataset(torch.utils.data.Dataset):

    def __init__(self, datasets, partition, options, **kwargs):
        """Load data from multiple datasets."""
        assert min(partition) >= 0
        assert sum(partition) == 1
        self.partition = np.array(partition).cumsum()
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in datasets]
        self.length = max(len(ds) for ds in self.datasets)

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
                return self.datasets[i][index_new]
        return None
