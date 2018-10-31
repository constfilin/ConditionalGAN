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
```  
(DeepLearning) cf@pfubuntu:ConditionalGAN$ python main.py --help

       USAGE: main.py [flags]
flags:

main.py:
  --batch_size: the batch size (the larger, the faster is training but if too large then model won't fit in memory and training is slow)
    (default: '64')
    (an integer)
  --data: data we are working with - mnist, celebA, wines
    (default: 'mnist')
  --data_path: location of the data on the file system
    (default: 'data')
  --generator_advantage: how many times run generator optimization per each run of discriminator optiomization. This helps prevent D loss going to 0
    (default: '2.0')
    (a number)
  --learn_rate: the learning rate for GAN
    (default: '0.0002')
    (a number)
  --log_path: the path of tensorflow's log
    (default: 'logs')
  --model_path: the folder where to save/restore the model
    (default: 'models')
  --operation: what are we going to be doing - train, test
    (default: 'test')
  --sample_path: the dir of sample images
    (default: 'samples')
  --samples: number of samples to generate
    (default: '1')
    (an integer)
  --samples_spec: samples specification - a comma separated list of one or many labels
    (default: 'random')
  --save_frequency: how often (in training steps) to save the model to place defined by 'model_path'
    (default: '50')
    (an integer)
  --[no]show_count_loss: should the program show less at each save point (if true the output is more verbose but training is slower
    (default: 'false')
  --training_steps: number of training steps
    (default: '10000')
    (an integer)
  --z_dim: the dimension of noise z
    (default: '100')
    (an integer)

Try --helpfull to get a list of all flags.
```

## TODO
On MNIST image set this Conditional GAN seems to work fine generating images of handwritten digits based on their labels. In other words, using `--operation=test` (i.e. "test") and a label (e.g. "7") it is possible to generate a random image of handwritten digit "7". 

I believe it works only because all the labels are mutually exclusive, i.e. it does not make sense to try to generate an image that matches (say) labels "7" and "3" at the same time.

The issue is that [this does not work](http://prntscr.com/lcjoxo) in case of [CelebA image set](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) where the attributes are not mutually exclusive. i.e. people might want to try using this Conditional GAN to generate images of celebrities with attributes like `Male`, `Young`, `Blond_Hair` and `Sideburns` at the same time. I.e. in this case the attributes are not mutually exclusive. Should the Conditional GAN work in this case?

## Reference code
[Conditional-GAN](https://github.com/zhangqianhui/Conditional-GAN/)

[DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
