#!/usr/bin/env python3
# coding: utf-8

# Standard imports
from typing import Optional, Tuple
from functools import reduce
import operator
# External imports
import torch
import torch.nn as nn


def conv_bn_leakyrelu(in_channels, out_channels):
    """
    Conv(3x3, same) - BN - LeakyRelu(0.3)
    """
    ks = 3
    return [
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=ks,
                  stride=1,
                  padding=int((ks-1)/2),
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.3)
    ]


def conv_downsampling(channels):
    """
    Conv(3x3, s2) - LeakyRelu(0.3)
    """
    ks = 3
    return [
        nn.Conv2d(channels, channels,
                  kernel_size=ks,
                  stride=2,
                  padding=int((ks-1)/2),
                  bias=True),
        nn.LeakyReLU(negative_slope=0.3)
    ]


class Discriminator(nn.Module):
    """
    The discriminator network tells if the input image is real or not
    The output logit is supposed to be high(-ly positive) for real images
    and low (highly negative) for fake images
    """

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 dropout: float,
                 base_c: int) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float) the probability of zeroing before the FC layer
            base_c (int): The base number of channels for the discriminator
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        in_C = img_shape[0]
        # Note: the output receptive field size is 36 x 36
        #       the output representation size is 3 x 3
        self.cnn = nn.Sequential(
            *conv_bn_leakyrelu(in_C, base_c),
            *conv_bn_leakyrelu(base_c, base_c),
            *conv_downsampling(base_c),
            nn.Dropout2d(dropout),
            *conv_bn_leakyrelu(base_c, base_c*2),
            *conv_bn_leakyrelu(base_c*2, base_c*2),
            *conv_downsampling(base_c*2),
            nn.Dropout2d(dropout),
            *conv_bn_leakyrelu(base_c*2, base_c*3),
            *conv_bn_leakyrelu(base_c*3, base_c*3),
            *conv_downsampling(base_c*3),
            nn.Dropout2d(dropout)
        )

        # Compute the size of the representation by forward propagating
        # a fake tensor; This can be cpu tensor as the model is not yet
        # built and therefore not yet transfered to the GPU
        fake_input = torch.zeros((1, *img_shape))
        out_cnn = self.cnn(fake_input)
        print(f"The output shape of the convolutional part of the "
              f"discriminator is {out_cnn.shape}")
        num_features = reduce(operator.mul, out_cnn.shape[1:])

        self.classif = nn.Sequential(
            nn.Linear(num_features, 1)
        )

        # Run the initialization script
        self.apply(self.init_weights)

    def init_weights(self, m):
        """
        Initialize the weights of the convolutional layers
        """

        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.fill_(0.)

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator
        Args:
            X(torch.Tensor (B, C, H, W)) : The images to classify

        Returns:
            Logits (torch.Tensor (B, )) : The logits
        """

        out_cnn = self.cnn(X)
        input_classif = out_cnn.view((out_cnn.shape[0], -1))
        out_classif = self.classif(input_classif)
        return out_classif.squeeze()


def up_conv_bn_relu(in_channels, out_channels):
    """
    Upsampling with Upsample - Conv
    UpSample(x2) - Conv(3x3) - BN - Relu - Conv(3x3) - BN - Relu
    """
    ks = 3
    return [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=ks,
                  stride=1,
                  padding=int((ks-1)/2),
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels,
                  out_channels,
                  kernel_size=ks,
                  stride=1,
                  padding=int((ks-1)/2),
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]


def tconv_bn_relu(in_channels, out_channels, ksize, stride, pad, opad):
    """
    Upsampling with transposed convolutions
    TConv2D - BN - LeakyRelu(0.3)
    """
    return [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=ksize,
                               stride=stride,
                               padding=pad,
                               output_padding=opad),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.3)
    ]


class Generator(nn.Module):
    """
    The generator network generates image from random inputs
    """

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 latent_size: int,
                 base_c: int) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            latent_size (int) : The dimension of the latent space
            base_c (int) : The base number of channels
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.base_c = base_c

        self.upscale = nn.Sequential(
            nn.Linear(self.latent_size, 7*7*self.base_c*4, bias=False),
            nn.BatchNorm1d(7*7*self.base_c*4),
            nn.LeakyReLU(negative_slope=0.3)
        )

        self.model = nn.Sequential(
            *up_conv_bn_relu(self.base_c*4, self.base_c*2),
            *up_conv_bn_relu(self.base_c*2, self.base_c),
            nn.Conv2d(self.base_c, self.img_shape[0], 
                      kernel_size=1,stride=1, padding=0, bias=True),
            nn.Tanh()
        )

        # Note : size, stride, pad, opad
        # self.model = nn.Sequential(
        #     *tconv_bn_relu2(base_c*4, base_c*2, 5, 1, 2, 0),
        #     # nn.Dropout2d(0.3),
        #     *tconv_bn_relu2(base_c*2, base_c, 5, 2, 2, 1),
        #     # nn.Dropout2d(0.3),
        #     nn.ConvTranspose2d(base_c, 1, 5, 2, 2, 1),
        #     nn.Tanh()  # as suggested by [Radford, 2016]
        # )

        # Initialize the convolutional layers
        self.apply(self.init_weights)

    def init_weights(self, m):
        with torch.no_grad():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    m.bias.fill_(0.)

    def forward(self,
                X: Optional[torch.Tensor] = None,
                batch_size: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of the generator. You can either provide a noise
        input vector or specify the batch_size to let it generate the input

        Args:
            X (torch.Tensor, optional): The input noise batch
            batch_size (int, optional): The number of samples to generate
        """
        # X is expected to be a 2D tensor (B, L)
        if X is None:
            assert(batch_size is not None)
            device = next(self.parameters()).device
            X = torch.randn(batch_size, self.latent_size).to(device)
        else:
            if len(X.shape) != 2:
                raise RuntimeError("Expected a 2D tensor as input to the "
                                   f" generator got a {len(X.shape)}D tensor.")

        upscaled = self.upscale(X)
        X = upscaled.view(-1, self.base_c*4, 7, 7)
        out = self.model(X)

        return out


class GAN(nn.Module):

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 dropout: float,
                 discriminator_base_c: int,
                 latent_size: int,
                 generator_base_c: int) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float): The probability of zeroing before the FC layers
            discriminator_base_c (int) : The base number of channels for 
                                         the discriminator
            latent_size (int) : The size of the latent space for the generator
            generator_base_c (int) : The base number of channels for the
                                     generator
        """
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.discriminator = Discriminator(img_shape,
                                           dropout,
                                           discriminator_base_c)
        self.generator = Generator(img_shape,
                                   latent_size,
                                   generator_base_c)

    def forward(self,
                X: Optional[torch.Tensor],
                batch_size: Optional[float]):
        """
        Given true images, returns the generated tensors
        and the logits of the discriminator for both the generated tensors
        and the true tensors

        Args:
            X (torch.Tensor) : a real image or None if we just
                               want the logits for the generated images
            batch_size (int) : the batch to consider when generating
                               fake images
        """

        if X is None and batch_size is None:
            raise RuntimeError("Not both X and batch_size can be None")
        if X is not None and batch_size is not None:
            raise RuntimeError("Not both X and batch_size can be not None")

        if X is not None:
            real_logits = self.discriminator(X)
            return real_logits, X
        else:
            fake_images = self.generator(X=None, batch_size=batch_size)
            fake_logits = self.discriminator(fake_images)

            return fake_logits, fake_images


def test_tconv():
    layers = nn.Sequential(
        nn.Conv2d(20, 10, kernel_size=3, stride=1, padding=2)
    )
    print(layers)
    inputs = torch.zeros((1, 20, 2, 2))
    outputs = layers(inputs)
    print(outputs.shape)

    imagify = nn.Linear(100, 7*7*10)
    conv1 = nn.ConvTranspose2d(10, 10,
                               kernel_size=5,
                               stride=1,
                               padding=2)
    conv2 = nn.ConvTranspose2d(10, 10,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1)
    conv3 = nn.ConvTranspose2d(10, 1,
                               kernel_size=5,
                               stride=2,
                               padding=2, output_padding=1)

    X = torch.randn(64, 100)
    X = imagify(X).view(-1, 10, 7, 7)
    print('--')
    print(X.shape)
    X = conv1(X)
    print(X.shape)
    X = conv2(X)
    print(X.shape)
    X = conv3(X)
    print(X.shape)



if __name__ == '__main__':
    test_tconv()
