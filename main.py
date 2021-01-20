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


# def train_gan():
#     pos_labels = torch.ones((bi, )).to(device)
#     neg_labels = torch.zeros((bi, )).to(device)

#     # Forward pass for training the discriminator
#     real_logits, fake_logits, _ = model(X, None)

#     Dloss = loss(real_logits, pos_labels) + \
#             loss(fake_logits, neg_labels)
#     dloss_e = Dloss.item()

#     optim_discriminator.zero_grad()
#     Dloss.backward()
#     optim_discriminator.step()

#     # Forward pass for training the generator
#     optim_generator.zero_grad()
#     _, fake_logits, _ = model(None, batch_size=bi)

#     # The generator wants his generated images to be positive
#     Gloss = loss(fake_logits, pos_labels)
#     gloss_e = Gloss.item()

#     optim_generator.zero_grad()
#     Gloss.backward()
#     optim_generator.step()


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
    nc = args.ncritic
    clip = args.clip

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
    critic = model.discriminator
    generator = model.generator
    optim_critic = optim.RMSprop(critic.parameters(),
                                 lr=base_lr)
    optim_generator = optim.RMSprop(generator.parameters(),
                                    lr=base_lr)

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

    # Generate few samples from the generator
    model.eval()
    _, _, fake_images = model(None, batch_size=16)
    grid = torchvision.utils.make_grid(fake_images,
                                       nrow=4,
                                       normalize=True)
    tensorboard_writer.add_image("Generated", grid, 0)
    torchvision.utils.save_image(grid, 'images.png')

    # Training loop
    for e in range(num_epochs):

        tot_closs = tot_gloss = 0
        Nc = Ng = 0
        model.train()
        for ei, (X, _) in enumerate(tqdm.tqdm(train_loader)):

            # X is a batch of real data
            X = X.to(device)
            bi = X.shape[0]

            # Optimize the critic
            real_values, fake_values, _ = model(X, None)
            critic_loss = -(real_values.mean() - fake_values.mean())
            optim_critic.zero_grad()
            critic_loss.backward()
            optim_critic.step()

            # Clip the weights of the critic
            with torch.no_grad():
                for p in critic.parameters():
                    p.clip_(clip)

            # Optimize the generator
            if ei % nc == 0:
                _, fake_values, _ = model(None, batch_size)
                generator_loss = -fake_values.mean()
                optim_generator.zero_grad()
                generator_loss.backward()
                optim_generator.step()

            Nc += 2*bi
            tot_closs += (bi * critic_loss.item())
            Ng += bi
            tot_gloss += (bi * generator_loss.item())
        tot_closs /= Nc
        tot_gloss /= Ng
        print(f"C loss : {tot_closs} ; G loss : {tot_gloss}")

        tensorboard_writer.add_scalar("Critic loss", tot_closs, e+1)
        tensorboard_writer.add_scalar("Generator loss", tot_gloss, e+1)

        # Generate few samples from the generator
        model.eval()
        _, _, fake_images = model(None, batch_size=16)
        # Unscale the images
        fake_images = fake_images * data._MNIST_STD + data._MNIST_MEAN
        grid = torchvision.utils.make_grid(fake_images,
                                           nrow=4,
                                           normalize=True,
                                           range=(-1, 1))
        tensorboard_writer.add_image("Generated", grid, e+1)


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
                        default=64)
    parser.add_argument("--base_lr",
                        type=float,
                        help="The initial learning rate to use",
                        default=0.00005)
    parser.add_argument("--debug",
                        action="store_true",
                        help="Whether to use small datasets")

    parser.add_argument("--ncritic",
                        type=int,
                        help="The number of batches for training the critic"
                             " before making one update if the generator",
                        default=5)
    parser.add_argument("--clip",
                        type=float,
                        help="The clipping value for the weights "
                             "of the critic",
                        default=0.01)
    # Regularization
    parser.add_argument("--dropout",
                        type=float,
                        help="The probability of zeroing before the FC layers",
                        default=0.3)

    args = parser.parse_args()

    eval(f"{args.command}(args)")
