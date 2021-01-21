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
    discriminator_base_c = args.discriminator_base_c
    generator_base_c = args.generator_base_c
    latent_size = args.latent_size
    sample_nrows = 8
    sample_ncols = 8

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
    model = models.GAN(img_shape,
                       dropout,
                       discriminator_base_c,
                       latent_size,
                       generator_base_c)
    model.to(device)

    # Optimizers
    critic = model.discriminator
    generator = model.generator
    optim_critic = optim.Adam(critic.parameters(),
                              lr=base_lr)
    optim_generator = optim.Adam(generator.parameters(),
                                 lr=base_lr)

    loss = torch.nn.BCEWithLogitsLoss()

    # Callbacks
    summary_text = "## Summary of the model architecture\n" + \
            f"{deepcs.display.torch_summarize(model)}\n"
    summary_text += "\n\n## Executed command :\n" +\
        "{}".format(" ".join(sys.argv))
    summary_text += "\n\n## Args : \n {}".format(args)

    logger.info(summary_text)

    logdir = generate_unique_logpath('./logs', 'gan')
    tensorboard_writer = SummaryWriter(log_dir = logdir,
                                       flush_secs=5)
    tensorboard_writer.add_text("Experiment summary", deepcs.display.htmlize(summary_text))

    with open(os.path.join(logdir, "summary.txt"), 'w') as f:
        f.write(summary_text)

    save_path = os.path.join(logdir, 'generator.pt')

    logger.info(f">>>>> Results saved in {logdir}")

    # Define a fixed noise used for sampling
    fixed_noise = torch.randn(sample_nrows*sample_ncols,
                              latent_size).to(device)

    # Generate few samples from the initial generator
    model.eval()
    fake_images = model.generator(X=fixed_noise)
    grid = torchvision.utils.make_grid(fake_images,
                                       nrow=sample_nrows,
                                       normalize=True)
    tensorboard_writer.add_image("Generated", grid, 0)
    torchvision.utils.save_image(grid, 'images/images-0000.png')

    # Training loop
    for e in range(num_epochs):

        tot_closs = tot_gloss = 0
        Nc = Ng = 0
        model.train()
        for ei, (X, _) in enumerate(tqdm.tqdm(train_loader)):

            # X is a batch of real data
            X = X.to(device)
            bi = X.shape[0]

            pos_labels = torch.ones((bi, )).to(device)
            neg_labels = torch.zeros((bi, )).to(device)

            # Forward pass for training the discriminator
            real_logits, _ = model(X, None)
            fake_logits, _ = model(None, bi)

            Dloss = loss(real_logits, pos_labels) + \
                    loss(fake_logits, neg_labels)
            dloss_e = Dloss.item()

            optim_critic.zero_grad()
            Dloss.backward()
            optim_critic.step()

            # Forward pass for training the generator
            optim_generator.zero_grad()
            fake_logits, _ = model(None, bi)

            # The generator wants his generated images to be positive
            Gloss = loss(fake_logits, pos_labels)
            gloss_e = Gloss.item()

            optim_generator.zero_grad()
            Gloss.backward()
            optim_generator.step()

            Nc += 2*bi
            tot_closs += 2 * bi * dloss_e
            Ng += bi
            tot_gloss += bi * gloss_e

        tot_closs /= Nc
        tot_gloss /= Ng
        logger.info(f"[Epoch {e+1}] C loss : {tot_closs} ; G loss : {tot_gloss}")

        tensorboard_writer.add_scalar("Critic loss", tot_closs, e+1)
        tensorboard_writer.add_scalar("Generator loss", tot_gloss, e+1)

        # Generate few samples from the generator
        model.eval()
        fake_images = model.generator(X=fixed_noise)
        # Unscale the images
        fake_images = fake_images * data._MNIST_STD + data._MNIST_MEAN
        grid = torchvision.utils.make_grid(fake_images,
                                           nrow=sample_nrows,
                                           normalize=True)
        tensorboard_writer.add_image("Generated", grid, e+1)
        torchvision.utils.save_image(grid, f'images/images-{e+1:04d}.png')

        real_images = X[:sample_nrows*sample_ncols,...]
        X = X * data._MNIST_STD + data._MNIST_MEAN
        grid = torchvision.utils.make_grid(real_images,
                                           nrow=sample_nrows,
                                           normalize=True)
        tensorboard_writer.add_image("Real", grid, e+1)

        # We save the generator
        logger.info(f"Generator saved at {save_path}")
        torch.save(model.generator, save_path)


def generate(args):
    """
    Function to generate new samples from the generator
    using a pretrained network
    """

    # Parameters
    modelpath = args.modelpath
    assert(modelpath is not None)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    
    # Reload the generator
    generator = torch.load(modelpath).to(device)
    generator.eval()

    # Generate some samples
    # sample_nrows = 8
    # sample_ncols = 8
    # z = torch.randn(sample_nrows * sample_ncols,
    #                 generator.latent_size).to(device)

    # fake_images = generator(z)
    # fake_images = fake_images * data._MNIST_STD + data._MNIST_MEAN
    # grid = torchvision.utils.make_grid(fake_images,
    #                                    nrow=sample_nrows,
    #                                    normalize=True)
    # torchvision.utils.save_image(grid, f'generated.png')


    # Interpolate in the laten space
    N = 20
    z = torch.zeros((N, N, generator.latent_size)).to(device)
    # Generate the 3 corner samples
    z[0, 0, :] = torch.randn(generator.latent_size)
    z[-1, 0, :] = torch.randn(generator.latent_size)
    z[0, -1, :] = torch.randn(generator.latent_size)
    di = z[-1, 0, :] - z[0, 0, :]
    dj = z[0, -1, :] - z[0, 0, :]
    for i in range(0, N):
        for j in range(0, N):
            z[i, j, :] = z[0, 0, :] + i/(N-1) * di + j/(N-1)*dj
    fake_images = generator(z.reshape(N**2, -1))
    fake_images = fake_images * data._MNIST_STD + data._MNIST_MEAN
    grid = torchvision.utils.make_grid(fake_images,
                                       nrow=N,
                                       normalize=True)
    torchvision.utils.save_image(grid, f'generated.png')



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
                        default=100)
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

    # Architectures
    parser.add_argument("--discriminator_base_c",
                        type=int,
                        help="The base number of channels for the discriminator",
                        default=32)
    parser.add_argument("--generator_base_c",
                        type=int,
                        help="The base number of channels for the generator",
                        default=64)
    parser.add_argument("--latent_size",
                        type=int,
                        help="The dimension of the latent space",
                        default=100)

    # Regularization
    parser.add_argument("--dropout",
                        type=float,
                        help="The probability of zeroing before the FC layers",
                        default=0.3)

    # For the generation
    parser.add_argument("--modelpath",
                        type=str,
                        help="The path to the pt file of the generator to load",
                        default=None)
    
    args = parser.parse_args()

    eval(f"{args.command}(args)")
