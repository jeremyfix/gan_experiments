#!/usr/bin/env python3
# coding: utf-8

# Standard imports
from typing import Optional, Tuple
# External imports
import torch
import torch.nn as nn


class GAN(nn.Module):

    def __init__(self,
                 img_shape: Tuple[int, int, int]) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
        """
        super(GAN, self).__init__()
        self.img_shape = img_shape
        self.discriminator = Discriminator(img_shape)
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
                 img_shape: Tuple[int, int, int]) -> None:
        """
        Args:
            img_shape : (C, H, W) image shapes
        """
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.cnn = nn.Sequential(

        )

    def forward(self,
                X: torch.Tensor) -> torch.Tensor:
        pass


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
