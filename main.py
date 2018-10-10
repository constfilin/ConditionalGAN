import os
import tensorflow as tf

from Data       import MnistData
from MnistModel import ConditionalGAN

flags = tf.app.flags
flags.DEFINE_string ("operation"  , "test"          , "what are we going to be doing - train, test, visualize")
flags.DEFINE_string ("data"       , "mnist"         , "data we are working with - mnist, celebA, lsun")
flags.DEFINE_integer("batch_size" , 64              , "the batch number")
flags.DEFINE_integer("z_dim"      , 100             , "the dimension of noise z")
flags.DEFINE_integer("y_dim"      , 10              , "the dimension of condition y")
flags.DEFINE_string ("model_path" , "models"        , "the path of model")
flags.DEFINE_string ("log_path"   , "logs"          , "the path of tensorflow's log")
flags.DEFINE_string ("sample_path", "samples"       , "the dir of sample images")
flags.DEFINE_integer("sample_cnt" , 1               , "number of samples to generate")
flags.DEFINE_float  ("learn_rate" , 0.0002          , "the learning rate for gan")
flags.DEFINE_string ("visual_path", "visualization" , "the path of visuzation images")

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
    
    if flags.FLAGS.data=="mnist":
        data = MnistData()
    elif flags.FLAGS.data=="celebA":
        data = CelebAData()
    else:
        raise ValueError("Data %s is not supported" % (flags.FLAGS.data))

    cgan = ConditionalGAN(data,flags.FLAGS.batch_size,flags.FLAGS.z_dim,flags.FLAGS.y_dim)

    # Model, logs, samples and visuals are all going to be saved in the folder dependent on the name of the data (i.e. mnist, celebA, lsun etc)
    [model_path,log_path,sample_path,visual_path] = append_to_paths(data.name,flags.FLAGS.model_path,flags.FLAGS.log_path,flags.FLAGS.sample_path,flags.FLAGS.visual_path)
    save_path = os.path.join(model_path,"model")

    if flags.FLAGS.operation == "train":
        make_paths(model_path,log_path,sample_path)
        cgan.train(save_path,log_path,sample_path,flags.FLAGS.learn_rate)
    elif flags.FLAGS.operation == "test":
        make_paths(sample_path)
        cgan.test(save_path,sample_path,flags.FLAGS.sample_cnt)
    elif flags.FLAGS.operation == "visualize":
        make_paths(visual_path)
        cgan.visual(save_path,visual_path)
    else:
        print("Unknown operation %s" % (flags.FLAGS.operation))

if __name__ == '__main__':
    tf.app.run()
