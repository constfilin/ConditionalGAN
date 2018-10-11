# Conditional Gans
The test code for Conditional Generative Adversarial Nets using tensorflow.

## INTRODUCTION

Tensorflow implements of [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784).
The paper must be the first one to introduce Conditional GANS but it did not provide source codes.

One attempt to create source code for CGAN was [made by zhangqianhui](https://github.com/zhangqianhui/Conditional-GAN/).
This code takes advantage of this prior work and takes it further by:
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
  Train & Test:
```  
    (DeepLearning) cf@cfdell:ConditionalGAN$ python main.py --help

       USAGE: main.py [flags]
flags:

main.py:
  --batch_size: the batch number
    (default: '64')
    (an integer)
  --data: data we are working with - mnist, celebA, lsun
    (default: 'mnist')
  --data_path: location of the data on the file system
    (default: 'data')
  --learn_rate: the learning rate for gan
    (default: '0.0002')
    (a number)
  --log_path: the path of tensorflow's log
    (default: 'logs')
  --model_path: the folder where to save/restore the model
    (default: 'models')
  --operation: what are we going to be doing - train, test
    (default: 'test')
  --sample_cnt: number of samples to generate
    (default: '1')
    (an integer)
  --sample_path: the dir of sample images
    (default: 'samples')
  --z_dim: the dimension of noise z
    (default: '100')
    (an integer)

Try --helpfull to get a list of all flags.
```
## Reference code
[Conditional-GAN](https://github.com/zhangqianhui/Conditional-GAN/)

[DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
