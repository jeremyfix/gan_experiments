#!/usr/bin/env python3

# Standard imports
import argparse
import logging
# External imports
import torch


def train(args):
    """
    Training of the algorithm
    """
    logger = logging.getLogger(__name__)
    logger.info("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # Dataloaders

    # Model definition

    # Training loop


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
    parser.add_argument("--dataset",
                        choices=["MNIST"],
                        help="Which dataset to use")
    parser.add_argument("--nthreads",
                       type=int,
                       help="The number of threads to use for loading the data",
                       default=6)

    args = parser.parse_args()

    eval(f"{args.command}(args)")

