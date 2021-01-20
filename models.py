#!/usr/bin/env python3
# coding: utf-8

# Standard imports
from typing import Optional, Tuple
from functools import reduce
import operator
# External imports
import torch
import torch.nn as nn


def bn_dropout_linear(dim_in, dim_out, p_drop):
    return [nn.BatchNorm1d(dim_in),
            nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out)]


def dropout_linear(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out)]


def dropout_linear_relu(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out),
            nn.ReLU()]


def conv_bn_leakyrelu(in_channels, out_channels):
    """
    Conv - BN - Relu
    """
    ks = 3
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2),
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)]


def conv_downsampling(channels):
    """
    Conv stride 2
    """
    ks = 3
    return [nn.Conv2d(channels, channels,
                      kernel_size=ks,
                      stride=2,
                      padding=int((ks-1)/2),
                      bias=True),
            nn.LeakyReLU(negative_slope=0.2)]


class Discriminator(nn.Module):
    """
    The discriminator network tells if the input image is real or not
    The output logit is supposed to be high(-ly positive) for real images
    and low (highly negative) for fake images
    """

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 dropout: float) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float) the probability of zeroing before the FC layer
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        in_C = img_shape[0]
        # Note: the output receptive field size is 36 x 36
        #       the output representation size is 3 x 3
        self.cnn = nn.Sequential(
            *conv_bn_leakyrelu(in_C, 32),
            *conv_bn_leakyrelu(32, 32),
            *conv_downsampling(32),
            # nn.Dropout2d(dropout),
            *conv_bn_leakyrelu(32, 64),
            *conv_bn_leakyrelu(64, 64),
            *conv_downsampling(64),
            # nn.Dropout2d(dropout),
            *conv_bn_leakyrelu(64, 128),
            *conv_bn_leakyrelu(128, 128),
            *conv_downsampling(128)
            # nn.Dropout2d(dropout)
        )

        # Compute the size of the representation by forward propagating
        # a fake tensor; This can be cpu tensor as the model is not yet
        # built and therefore not yet transfered to the GPU
        fake_input = torch.zeros((1, *img_shape))
        out_cnn = self.cnn(fake_input)
        num_features = reduce(operator.mul, out_cnn.shape[1:])

        self.classif = nn.Sequential(
            nn.Linear(num_features, 1)
        )

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:
        out_cnn = self.cnn(X)
        input_classif = out_cnn.view((out_cnn.shape[0], -1))
        out_classif = self.classif(input_classif)
        return out_classif.squeeze()


def up_conv_bn_relu(in_channels, out_channels):
    ks = 3
    return [
        nn.Upsample(scale_factor=2),
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel_size=ks,
                  stride=1,
                  padding=int((ks-1)/2),
                  bias=True),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.2)
    ]


def tconv_bn_relu(in_channels, out_channels,
                  ksize, stride, pad, opad):
    return [
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=ksize,
                               stride=stride,
                               padding=pad,
                               output_padding=opad,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
    ]


class Generator(nn.Module):
    """
    The generator network generates image from random inputs
    """

    def __init__(self,
                 img_shape: Tuple[int, int, int]) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
        """
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_size = 100

        base_c = 64
        self.upscale = nn.Sequential(
            nn.Linear(self.latent_size, 7*7*256, bias=False),
            nn.BatchNorm1d(7*7*256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Note : size, stride, pad, opad
        self.model = nn.Sequential(
            *up_conv_bn_relu(256, 64),
            *up_conv_bn_relu(64, 1)
        )
        # self.model = nn.Sequential(
        #     *tconv_bn_relu(base_c*4, base_c*2, 6, 2, 2, 0),
        #     *tconv_bn_relu(base_c*2, self.img_shape[0], 6, 2, 2, 0),
        #     nn.Tanh()  # as suggested by [Radford, 2016]
        # )

    def forward(self,
                X: Optional[torch.Tensor] = None,
                batch_size: Optional[float] = None) -> torch.Tensor:
        """
        Forward pass of the generator. You can either provide a noise
        input vector or specify the batch_size to let it generate the input
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
        # Make X a 1x1 image like tensor

        upscaled = self.upscale(X)
        upscaled = upscaled.view(-1, 256, 7, 7)
        out = self.model(upscaled)

        return out

class GAN(nn.Module):

    def __init__(self,
                 img_shape: Tuple[int, int, int],
                 dropout: float) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
            dropout (float): The probability of zeroing before the FC layers
        """
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.discriminator = Discriminator(img_shape, dropout)
        self.generator = Generator(img_shape)

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
        """

        if X is None and batch_size is None:
            raise RuntimeError("Not both X and batch_size can be None")

        if X is not None:
            real_logits = self.discriminator(X)
            batch_size = X.shape[0]
        else:
            real_logits = None
            batch_size = batch_size

        fake_images = self.generator(batch_size=batch_size)
        fake_logits = self.discriminator(fake_images)

        return real_logits, fake_logits, fake_images


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
                               kernel_size=6,
                               stride=2,
                               padding=2)
    conv2 = nn.ConvTranspose2d(10, 1,
                               kernel_size=6,
                               stride=2,
                               padding=2)

    X = torch.randn(64, 100)
    X = imagify(X).view(-1, 10, 7, 7)
    print(X.shape)
    X = conv1(X)
    print(X.shape)
    X = conv2(X)
    print(X.shape)


if __name__ == '__main__':
    test_tconv()
