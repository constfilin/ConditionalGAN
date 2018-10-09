# Conditional-Gans
The test code for Conditional Generative Adversarial Nets using tensorflow.

## INTRODUCTION

Tensorflow implements of [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).
The paper should be the first one to introduce Conditional GANS.
But they did not provide source codes.

This code takes advantage of the [prior work by zhangqianhui](https://github.com/zhangqianhui/Conditional-GAN/) and in addition:
1. Works with Python 3.X
1. Eliminates numerous hardcodings
1. Easier to customize beyond MNIST Conditional GAN
1. Written by a software engineer and for software engineers :)

## Prerequisites
- tensorflow >=1.0
- python 3.X

## Usage

  Download mnist:
  
    $ python download.py mnist
  
  Train:
  
    $ python main.py --operation train
  
  Test:
  
    $ python main.py --operation test
  
  Visualization:
  
    $ python main.py --operation visualize

## Reference code
[Conditional-GAN](https://github.com/zhangqianhui/Conditional-GAN/)
[DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
