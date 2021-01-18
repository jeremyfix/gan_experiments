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

    def forward(self, X):
        """
        Given true images, returns the generated tensors
        and the logits of the discriminator for both the generated tensors
        and the true tensors
        """

        # Step 1
        generated_images = self.generator()

        positive_logits = self.discriminator(X)
        negative_logits = self.discriminator(X)

        return generated_images, positive_logits, negative_logits


class Discriminator(nn.Module):
    """
    The discriminator network tells if the input image is a fake or not
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
            *conv_bn_leakyrelu(32, 32),
            *conv_bn_leakyrelu(32, 32),
            *conv_downsampling(32),
            *conv_bn_leakyrelu(32, 64),
            *conv_bn_leakyrelu(64, 64),
            *conv_downsampling(64)
        )

        # Compute the size of the representation by forward propagating
        # a fake tensor; This can be cpu tensor as the model is not yet
        # built and therefore not yet transfered to the GPU
        fake_input = torch.zeros((1, *img_shape))
        out_cnn = self.cnn(fake_input)
        num_features = reduce(operator.mul, out_cnn.shape[1:])

        self.classif = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 1)
        )

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:
        out_cnn = self.cnn(X)
        input_classif = out_cnn.view((out_cnn.shape[0], -1))
        out_classif = self.classif(input_classif)
        return out_classif


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

    def forward(self,
                X: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
