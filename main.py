import os
import tensorflow as tf

from MnistData  import MnistData
from MnistModel import ConditionalGAN

flags = tf.app.flags
flags.DEFINE_string ("operation"  , "test"            , "what are we going to be doing - train, test, visualize")
flags.DEFINE_integer("batch_size" , 64                , "the batch number")
flags.DEFINE_integer("z_dim"      , 100               , "the dimension of noise z")
flags.DEFINE_integer("y_dim"      , 10                , "the dimension of condition y")
flags.DEFINE_string ("model_path" , "model/model.ckpt", "the path of model")
flags.DEFINE_string ("log_path"   , "tf_log"          , "the path of tensorflow's log")
flags.DEFINE_string ("sample_path", "samples"         , "the dir of sample images")
flags.DEFINE_float  ("learn_rate" , 0.0002            , "the learning rate for gan")
flags.DEFINE_string ("visual_path", "visualization"   , "the path of visuzation images")

def makt_paths(*paths):
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p);

def main(_):
    cg = ConditionalGAN(MnistData(),flags.FLAGS.batch_size,flags.FLAGS.z_dim,flags.FLAGS.y_dim)
    if flags.FLAGS.operation == "train":
        make_paths(flags.FLAGS.model_path,flags.FLAGS.log_path,flags.FLAGS.sample_path)
        cg.train(flags.FLAGS.model_path,flags.FLAGS.log_path,flags.FLAGS.sample_path,flags.FLAGS.learn_rate)
    elif flags.FLAGS.operation == "test":
        make_paths(flags.FLAGS.sample_path)
        cg.test(flags.FLAGS.model_path,flags.FLAGS.sample_path)
    elif flags.FLAGS.operation == "visualize":
        make_paths(flags.FLAGS.visual_path)
        cg.visual(flags.FLAGS.model_path,flags.FLAGS.visual_path)
    else:
        print("Unknown operation %s" % (flags.FLAGS.operation))

if __name__ == '__main__':
    tf.app.run()
