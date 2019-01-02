import os
import tensorflow as tf

import Data
import ConditionalGAN

flags = tf.app.flags
flags.DEFINE_string ("operation"          , "test"          , "what are we going to be doing - train, test")
flags.DEFINE_string ("data"               , "mnist"         , "data we are working with - mnist, celebA, wines")
flags.DEFINE_integer("z_dim"              , 100             , "the dimension of noise z")
flags.DEFINE_integer("batch_size"         , 64              , "the batch size (the larger, the faster is training but if too large then model won't fit in memory and training is slow)")
# Training arguments
flags.DEFINE_float  ("learn_rate"         , 0.0002          , "the learning rate for GAN")
flags.DEFINE_integer("training_steps"     , 10000           , "number of training steps")
flags.DEFINE_integer("save_frequency"     , 50              , "how often (in training steps) to save the model to place defined by 'model_path'")
flags.DEFINE_float  ("generator_advantage", 0.0             , "how many times run generator optimization per each run of discriminator optimization.\n"+
                     "Any value <=0 means dynamic adjustment of advantage to keep G and D loss about equal.\n"+
                     "This helps prevent D loss going to 0")
# Testing arguments
flags.DEFINE_integer("samples"            , 1               , "number of samples to generate")
flags.DEFINE_string ("samples_spec"       , "random"        , "samples specification - a comma separated list of one or many labels")
# These rarely need changing
flags.DEFINE_string ("data_path"          , "data"          , "location of the data on the file system")
flags.DEFINE_string ("model_path"         , "models"        , "the folder where to save/restore the model")
flags.DEFINE_string ("log_path"           , "logs"          , "the path of tensorflow's log")
flags.DEFINE_string ("sample_path"        , "samples"       , "the dir of sample images")

def append_to_paths(name,*paths):
    result = []
    for p in paths:
        result.append(os.path.join(p,name))
    return result

def make_paths(*paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p);
    return paths

def main(_):
    
    # Data, model, logs and samples are all going to be saved in the folder dependent on the name of the data (i.e. mnist, celebA, lsun etc)
    [data_path,model_path,log_path,sample_path] = append_to_paths(flags.FLAGS.data,
                                                                  flags.FLAGS.data_path,flags.FLAGS.model_path,flags.FLAGS.log_path,flags.FLAGS.sample_path)
    if flags.FLAGS.data=="mnist":
        data = Data.Mnist(data_path)
    elif flags.FLAGS.data=="banners":
        data = Data.Banners(data_path)
    elif flags.FLAGS.data=="artists":
        data = Data.Artists(data_path)
    elif flags.FLAGS.data=="celebA":
        data = Data.CelebA(data_path,5000)
    elif flags.FLAGS.data=="wines":
        data = Data.Wines(data_path)
    else:
        raise ValueError("Data %s is not supported" % (flags.FLAGS.data))

    cgan = ConditionalGAN.ConditionalGAN(data,flags.FLAGS.batch_size,flags.FLAGS.z_dim)

    if flags.FLAGS.operation == "train":
        make_paths(model_path,log_path,sample_path)
        cgan.train(model_path,log_path,sample_path,flags.FLAGS.training_steps,flags.FLAGS.learn_rate,flags.FLAGS.save_frequency,flags.FLAGS.generator_advantage)
    elif flags.FLAGS.operation == "test":
        make_paths(sample_path)
        if flags.FLAGS.samples_spec=="random":
            labels = data.get_random_labels((flags.FLAGS.samples,data.get_number_of_labels()))
        else:
            labels = data.get_labels_by_spec((flags.FLAGS.samples,data.get_number_of_labels()),flags.FLAGS.samples_spec)
        print("\n".join(data.describe_labels(labels)))
        cgan.test(model_path,sample_path,None,labels)
    else:
        print("Unknown operation %s" % (flags.FLAGS.operation))

if __name__ == '__main__':
    tf.app.run()
