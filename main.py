#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import argparse
import logging
import sys
# External imports
import torch
import deepcs
import deepcs.display
from deepcs.training import train as ftrain, ModelCheckpoint
from deepcs.testing import test as ftest
from deepcs.fileutils import generate_unique_logpath
import deepcs.metrics
# Local imports
import data
import models


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    # Parameters
    dataset_root = args.dataset_root
    nthreads = args.nthreads
    batch_size = args.batch_size
    dropout = args.dropout
    debug = args.debug

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Dataloaders
    train_loader, valid_loader, img_shape = data.get_dataloaders(dataset_root=dataset_root,
                                                                 cuda=use_cuda,
                                                                 batch_size=batch_size,
                                                                 n_threads = nthreads,
                                                                 dataset="MNIST",
                                                                small_experiment=debug)

    # Model definition
    model = models.GAN(img_shape, dropout)
    model.to(device)

    # Callbacks
    summary_text = "## Summary of the model architecture\n" + \
            f"{deepcs.display.torch_summarize(model)}\n"
    summary_text += "\n\n## Executed command :\n" +\
        "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    # Training loop

    X, _ = next(iter(train_loader))
    X = X.to(device)
    generated_images, positive_logits, negative_logits = model(X)


def generate(args):
    """
    Function to generate new samples from the generator
    using a pretrained network
    """
    pass


if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("command",
                        choices=['train', 'generate'])

    # Data parameters
    parser.add_argument("--dataset",
                        choices=["MNIST"],
                        help="Which dataset to use")
    parser.add_argument("--dataset_root",
                        type=str,
                        help="The root dir where the datasets are stored",
                        default=data._DEFAULT_DATASET_ROOT)
    parser.add_argument("--nthreads",
                        type=int,
                        help="The number of threads to use "
                        "for loading the data",
                        default=6)

    # Training parameters
    parser.add_argument("--batch_size",
                        type=int,
                        help="The size of a minibatch",
                        default=64)
    parser.add_argument("--debug",
                        action="store_true",
                        help="Whether to use small datasets")

    # Regularization
    parser.add_argument("--dropout",
                        type=float,
                        help="The probability of zeroing before the FC layers",
                        default=0.5)

    args = parser.parse_args()

    eval(f"{args.command}(args)")

