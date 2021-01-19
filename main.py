#!/usr/bin/env python3
# coding: utf-8

# Standard imports
import argparse
import logging
import sys
import os
# External imports
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import deepcs
import deepcs.display
from deepcs.training import train as ftrain, ModelCheckpoint
from deepcs.testing import test as ftest
from deepcs.fileutils import generate_unique_logpath
import deepcs.metrics
import tqdm
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
    base_lr = args.base_lr
    num_epochs = args.num_epochs

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

    # Optimizers
    optim_discriminator = optim.Adam(model.discriminator.parameters(),
                                     betas=[0.5, 0.999],
                                     lr=base_lr)
    optim_generator = optim.Adam(model.generator.parameters(),
                                 betas=[0.5, 0.999],
                                 lr=base_lr)

    loss = torch.nn.BCEWithLogitsLoss()

    # Callbacks
    summary_text = "## Summary of the model architecture\n" + \
            f"{deepcs.display.torch_summarize(model)}\n"
    summary_text += "\n\n## Executed command :\n" +\
        "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    logdir = generate_unique_logpath('./logs', 'ctc')
    tensorboard_writer = SummaryWriter(log_dir = logdir,
                                       flush_secs=5)
    tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

    with open(os.path.join(logdir, "summary.txt"), 'w') as f:
        f.write(summary_text)

    logger.info(f">>>>> Results saved in {logdir}")

    # Training loop
    for e in range(num_epochs):

        tot_dloss = tot_gloss = 0
        Ng = Nd = 0
        model.train()
        for X, _ in tqdm.tqdm(train_loader):

            X = X.to(device)
            bi = X.shape[0]

            pos_labels = torch.ones((bi, )).to(device)
            neg_labels = torch.zeros((bi, )).to(device)

            # Forward pass for training the discriminator
            real_logits, fake_logits, _ = model(X, None)

            Dloss = loss(real_logits, pos_labels) + \
                    loss(fake_logits, neg_labels)
            dloss_e = Dloss.item()

            optim_discriminator.zero_grad()
            Dloss.backward()
            optim_discriminator.step()

            # Forward pass for training the generator
            optim_generator.zero_grad()
            _, fake_logits, _ = model(None, batch_size=bi)

            # The generator wants his generated images to be positive
            Gloss = loss(fake_logits, pos_labels)
            gloss_e = Gloss.item()

            optim_generator.zero_grad()
            Gloss.backward()
            optim_generator.step()

            Nd += 2*bi
            tot_dloss += (batch_size * dloss_e)
            Ng += bi
            tot_gloss += (batch_size * gloss_e)

        print(f"D loss : {tot_dloss/Nd} ; G loss : {tot_gloss/Ng}")

        # Generate few samples from the generator
        model.eval()
        _, _, fake_images = model(None, batch_size=16)
        grid = torchvision.utils.make_grid(fake_images,
                                           nrow=4,
                                           normalize=True,
                                           range=(-1, 1))
        tensorboard_writer.add_image("Generated", grid, e)
        torchvision.utils.save_image(grid, 'images.png')







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
    parser.add_argument("--num_epochs",
                        type=int,
                        help="The number of epochs to train for",
                        default=50)
    parser.add_argument("--batch_size",
                        type=int,
                        help="The size of a minibatch",
                        default=128)
    parser.add_argument("--base_lr",
                        type=float,
                        help="The initial learning rate to use",
                        default=0.0001)
    parser.add_argument("--debug",
                        action="store_true",
                        help="Whether to use small datasets")

    # Regularization
    parser.add_argument("--dropout",
                        type=float,
                        help="The probability of zeroing before the FC layers",
                        default=0.3)

    args = parser.parse_args()

    eval(f"{args.command}(args)")
