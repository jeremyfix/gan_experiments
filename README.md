# GAN experiments

These are sample scripts for playing with DC-GAN. These scripts were developed for preparing a labwork on GAN (see [here](https://github.com/jeremyfix/deeplearning-lectures)). It can be used to train a DCGAN like architecture on MNIST, EMNIST, FashionMNIST or SVHN. On all four datasets, the generator seem to generate pretty realistic samples.

The discriminator is a convolutional network using stacked Conv3x3, with convolutional downsampling and batchnormalization and LeakyRelu. The generator is using the resize-convolution pattern with UpSampling followed by convolutional layers with batchnormalization and Relu and finally a Tanh. All is implemented with Pytorch and trained with pytorch-1.7.0 cuda 10.1. An epoch takes approximately 20s (for MNIST like) to 40s (for SVHN) on a 1080Ti.

## Experiments

### Training the GAN

Below we display a collection of images generated by the generator during 400 epochs. After 50 epochs, you already get
reasonnably good images but I wanted to see what is going on in the long term.

For training, just run 

```
python3 main.py train
```

![Fake digits generated during training](results/mnist.gif)

If you want to test with a different dataset, you can choose between MNIST, FashionMNIST, EMNIST or SVHN. Below is an example of generated images during training on SVHN.

![Fake house numbers generated during training](results/svhn.gif)

However, I'm really surprised by the losses and accuracies I get : a pretty low discriminator loss, a high discriminator accuracy and a high generator loss. All these metrics seem to indicate the generator is not doing a good job even if the generated digits are indeed, visually, reasonably good. 

![Losses and accuracies](results/metrics.png)


### Generating samples

You can generate new samples running : 

```
python3 main.py generate --modelpath generator.pt
```

Here are some examples of fake images using different random inputs.

![Fake house numbers at the end of training](results/svhn_samples.png)

### Interpolation experiment

We take three random vectors. The image of the generator are the fake digits, on the picture below, in the top-left, top-right and bottom-left corners. Then, for every cell, we compute an interpolate random input of which we display the image by the generator.

![The fake digits generated by the generator given three random seed](results/interpolated.png)

Below is an example on SVHN:

![The fake house numbers generated by the generator given three random seed](results/svhn_interpolated.png)

