#!/usr/bin/env python3

# Standard imports
from typing import Union
from pathlib import Path
# External imports
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

_DEFAULT_DATASET_ROOT = "/opt/Datasets"
_DEFAULT_MNIST_DIGIT = 6

def get_dataloaders(dataset_root: Union[str, Path],
                    cuda: bool,
                    batch_size: int = 64,
                    n_threads: int = 4,
                    dataset: str = "MNIST",
                    val_size: float = 0.2):
    """
    Build and return the pytorch dataloaders

    Args:
        dataset_root (str, Path) : the root path of the datasets
        cuda (bool): whether or not to use cuda
        batch_size (int) : the size of the minibatches
        n_threads (int): the number of threads to use for dataloading
        dataset (str): the dataset to load
        val_size (float): the proportion of data for the validation set
        
    """

    datasets = ["MNIST"]
    if not dataset in datasets:
        raise NotImplementedError(f"Cannot import the dataset {dataset}."
                                  f" Available datasets are {datasets}")

    if dataset == "MNIST":
        # Get the two datasets, make them tensors in [0, 1]
        train_dataset = torchvision.datasets.MNIST(root=dataset_root,
                                                   train=True,
                                                   download=True,
                                                   transform=transforms.ToTensor())
        test_dataset = torchvision.datasets.MNIST(root=dataset_root,
                                                  train=False,
                                                  download=True,
                                                  transform=transforms.ToTensor())
        dataset = torch.utils.data.ConcatDataset([train_dataset,
                                                 test_dataset])
        # And keep only the 6 from it
        idx_to_keep=[si for (si, (x,y)) in enumerate(dataset) if y == _DEFAULT_MNIST_DIGIT]
        dataset = torch.utils.data.Subset(dataset, idx_to_keep)

        # Split the dataset in train/valid
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        split_idx = int(val_size * len(dataset))
        valid_indices, train_indices = indices[:split_idx], indices[split_idx:]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        valid_dataset = torch.utils.data.Subset(dataset, valid_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_threads)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=n_threads)

        img_shape = dataset[0][0].shape # C, H, W

    return train_loader, valid_loader, img_shape


def test_mnist():
    import matplotlib.pyplot as plt

    train_loader, valid_loader, img_shape = get_dataloaders(dataset_root=_DEFAULT_DATASET_ROOT,
                                                            batch_size=16,
                                                            cuda=False,
                                                            dataset="MNIST")
    print(f"I loaded {len(train_loader)} train minibatches. The images"
          f" are of shape {img_shape}")

    X, y = next(iter(train_loader))

    grid = torchvision.utils.make_grid(X, nrow=4)
    print(grid.min(), grid.max())
    print(grid.shape)

    plt.figure()
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap='gray_r')
    plt.show()

if __name__ == '__main__':
    test_mnist()
