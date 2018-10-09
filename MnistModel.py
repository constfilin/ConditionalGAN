import tensorflow as tf
import numpy as np

from utils import *
from ops import *

def sample_label(shape):
    label_vector = np.zeros(shape,dtype=np.float)
    for i in range(0,shape[0]):
        label_vector[i,i//8] = 1.0
    return label_vector

class ConditionalGAN(object):

    def __init__(self,data,batch_size,z_dim,y_dim):
        self.data    = data
        self.images  = tf.placeholder(tf.float32, [batch_size,*self.data.shape], name="self.images")
        self.z       = tf.placeholder(tf.float32, [batch_size,z_dim], name="self.z")
        self.y       = tf.placeholder(tf.float32, [batch_size,y_dim], name="self.y")
        self.build_model()

    def build_model(self):
        self.fakes_generator     = self.get_generator_net(self.z,self.y)
        self.reals_discriminator = self.get_discriminator_net(self.images,self.y,False)
        self.fakes_discriminator = self.get_discriminator_net(self.fakes_generator[0],self.y,True)

        # TODO: define how the generator loss is counted
        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like (self.fakes_discriminator[0]) ,logits=self.fakes_discriminator[1]))
        # Discriminator loss - apparently - consists of loss in fake images that the discriminator will recognize as reals AND
        # the loss in real images that discriminator recognize as fakes
        discriminator_fakes_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.fakes_discriminator[0]) ,logits=self.fakes_discriminator[1]))
        discriminator_reals_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like (self.reals_discriminator[0]),logits=self.reals_discriminator[1]))
        self.discriminator_loss = discriminator_reals_loss+discriminator_fakes_loss

        self.saver   = tf.train.Saver()

    def train(self,model_path,log_path,sample_path,learning_rate):

        all_vars                = tf.trainable_variables()

        discriminator_vars      = [var for var in all_vars if var.name.startswith("discriminator/")]
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(self.discriminator_loss,var_list=discriminator_vars)
        discriminator_summary   = tf.summary.merge([
            tf.summary.scalar("discriminator_loss",self.discriminator_loss), 
            tf.summary.histogram("images_discriminator_pro",self.reals_discriminator[0])])

        generator_vars          = [var for var in all_vars if var.name.startswith("generator/")]
        generator_optimizer     = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(self.generator_loss,var_list=generator_vars)
        generator_summary       = tf.summary.merge([
            tf.summary.scalar("generator_loss",self.generator_loss), 
            tf.summary.histogram("fakes_discriminator_pro",self.fakes_discriminator[0]),
            tf.summary.image("fakes_generator_out",self.fakes_generator[0])])

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(log_path,graph=sess.graph)

            for step in range(0,10000):
                images, labels = self.data.get_next_batch(step,self.get_batch_size())

                # Get the z
                batch_z = np.random.uniform(-1,1,size=self.z.shape)

                _, summary_str = sess.run([discriminator_optimizer,discriminator_summary],feed_dict={
                    self.images: images, 
                    self.z     : batch_z, 
                    self.y     : labels})
                summary_writer.add_summary(summary_str,step)

                _, summary_str = sess.run([generator_optimizer,generator_summary],feed_dict={
                    self.z     : batch_z, 
                    self.y     : labels})
                summary_writer.add_summary(summary_str,step)

                if (step%50)==0:
                    discriminator_loss = sess.run(self.discriminator_loss, feed_dict={self.images: images, self.z: batch_z, self.y: labels})
                    generator_loss     = sess.run(self.generator_loss    , feed_dict={self.z: batch_z, self.y: labels})
                    print("Step %4d: D: loss=%.7f G: loss=%.7f " % (step,discriminator_loss,generator_loss))

                    sample_images      = sess.run(self.fakes_generator[0],feed_dict={self.z: batch_z, self.y: sample_label(self.y.shape)})
                    save_images(sample_images,[8,8],"./%s/train_%04d.png" % (sample_path,step))
                    self.saver.save(sess,model_path)

            save_path = self.saver.save(sess,model_path)
            print("Model saved in file: %s" % (save_path))

    def test(self,model_path,sample_path,count):
        def get_unique_filename( sample_path ):
            for i in range(0,10000):
                image_path = "./{}/test{:02d}_{:04d}.png".format(sample_path,0,i)
                if not os.path.isfile(image_path):
                    return image_path
            raise Exception("Cannot find unique file name in %s" % (sample_path))            
        np.random.seed()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess,model_path)
            for cnt in range(0,count):
                sample_z   = np.random.uniform(1,-1,size=self.z.shape)
                output     = sess.run(self.fakes_generator[0],feed_dict={self.z:sample_z,self.y:sample_label(self.y.shape)})
                image_path = get_unique_filename(sample_path)
                save_images(output,[8,8],image_path)
                print("A sample is saved in %s" % (image_path))
        print("Testing is done")

    def visual(self,model_path,visual_path):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess,model_path)

            # visualize the weights 1 or you can change weight_2 .
            conv_weights = sess.run([tf.get_collection('weight_2')])
            vis_square(visual_path,conv_weights[0][0].transpose(3, 0, 1, 2), type=1)

            # visualize the activation 2
            images,labels = self.data.get_next_batch(0,self.get_batch_size())
            ac = sess.run([tf.get_collection('ac_2')],feed_dict={
                self.images: images[:64], 
                self.z:      np.random.uniform(-1,1,size=self.z.shape), 
                self.y:      sample_label(self.y.shape)})
            vis_square(visual_path,ac[0][0].transpose(3, 1, 2, 0), type=0)
            print("the visualization finished.")

    def get_batch_size(self):
        return int(self.images.shape[0])

    def get_generator_net(self, z, y):
        with tf.variable_scope('generator') as scope:

            yb = tf.reshape(y,shape=[y.shape[0],1,1,-1])

            z = tf.concat([z, y], 1)

            d1 = tf.nn.relu(batch_normal(fully_connect(z,output_size=1024,scope='gen_fully'),scope='gen_bn1'))
            d1 = tf.concat([d1, y], 1)

            c1 = self.data.shape[0] // 4
            d2 = tf.nn.relu(batch_normal(fully_connect(d1,output_size=c1*c1*2*64,scope='gen_fully2'),scope='gen_bn2'))
            d2 = tf.reshape(d2, [self.get_batch_size(), c1, c1, 64 * 2])
            d2 = conv_cond_concat(d2, yb)

            c2 = self.data.shape[0] // 2
            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.get_batch_size(), c2, c2, 128], name='gen_deconv1'), scope='gen_bn3'))
            d3 = conv_cond_concat(d3, yb)

            d4 = de_conv(d3,output_shape=self.images.shape, name='gen_deconv2')

            return tf.nn.sigmoid(d4),d4

    def get_discriminator_net(self,images,y,reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                scope.reuse_variables()

            # mnist data's shape is (28 , 28 , 1)
            yb = tf.reshape(y,shape=[y.shape[0],1,1,-1])

            # concat
            concat_data = conv_cond_concat(images, yb)

            conv1, w1 = conv2d(concat_data,output_dim=10,name='dis_conv1')
            tf.add_to_collection('weight_1',w1)

            conv1 = lrelu(conv1)
            conv1 = conv_cond_concat(conv1,yb)
            tf.add_to_collection('ac_1',conv1)

            conv2, w2 = conv2d(conv1,output_dim=64,name='dis_conv2')
            tf.add_to_collection('weight_2',w2)

            conv2 = lrelu(batch_normal(conv2,scope='dis_bn1'))
            tf.add_to_collection('ac_2',conv2)

            conv2 = tf.reshape(conv2,[self.get_batch_size(),-1])
            conv2 = tf.concat([conv2,y],1)

            f1 = lrelu(batch_normal(fully_connect(conv2,output_size=1024,scope='dis_fully1'),scope='dis_bn2',reuse=reuse))
            f1 = tf.concat([f1,y],1)

            out = fully_connect(f1, output_size=1, scope='dis_fully2')

            return tf.nn.sigmoid(out),out
