import os
import re
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import xavier_initializer

from ops import *
from utils import *

class ConditionalGAN(object):

    def __init__(self,data,batch_size,z_dim):
        self.data    = data
        self.images = tf.placeholder(tf.float32,[batch_size,*data.shape], name="images")
        # TODO: figure out what significance the dimension z_dim has
        self.noise  = tf.placeholder(tf.float32,[batch_size,z_dim], name="noise")
        self.labels = tf.placeholder(tf.float32,[batch_size,data.get_number_of_labels()], name="labels")
        self.build_model()

    def build_model(self):
        # Here's the model we are building: http://prntscr.com/l46dnz
        self.fakes_generator     = self.get_generator_net("fakes_generator")
        self.reals_discriminator = self.get_discriminator_net("reals_discriminator",self.images,False)
        self.fakes_discriminator = self.get_discriminator_net("fakes_discriminator",self.fakes_generator[0],True)

        # Discriminator loss - apparently - consists of loss in fake images that the discriminator will recognize as reals AND
        # the loss in real images that discriminator recognize as fakes
        discriminator_fakes_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.fakes_discriminator[0]) ,logits=self.fakes_discriminator[1]))
        discriminator_reals_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like (self.reals_discriminator[0]),logits=self.reals_discriminator[1]))
        self.discriminator_loss  = tf.add(discriminator_reals_loss,discriminator_fakes_loss,name="discriminator_loss")
        # Generator produces a loss when the discriminator successfuly recognizes generated images as fakes
        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like (self.fakes_discriminator[0]) ,logits=self.fakes_discriminator[1]),name="generator_loss")

        self.saver   = tf.train.Saver(max_to_keep=5)

    def train(self,model_path,log_path,sample_path,training_steps,learning_rate,save_frequency,show_count_loss,generator_advantage):

        all_vars                = tf.trainable_variables()

        discriminator_summary   = tf.summary.merge([
            tf.summary.scalar("discriminator_loss",self.discriminator_loss), 
            tf.summary.histogram("images_discriminator_pro",self.reals_discriminator[0])])
        discriminator_vars      = [var for var in all_vars if var.name.startswith("discriminator/")]
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5,name="discriminator_optimizer").minimize(self.discriminator_loss,var_list=discriminator_vars)

        generator_summary       = tf.summary.merge([
            tf.summary.scalar("generator_loss",self.generator_loss), 
            tf.summary.histogram("fakes_discriminator_pro",self.fakes_discriminator[0]),
            tf.summary.image("fakes_generator_out",self.fakes_generator[0])])
        generator_vars          = [var for var in all_vars if var.name.startswith("generator/")]
        generator_optimizer     = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5,name="generator_optimizer").minimize(self.generator_loss,var_list=generator_vars)

        np.random.seed()
        samples_input_noise  = np.random.uniform(1,-1,size=self.noise.shape)
        samples_input_labels = self.data.get_random_labels(self.labels.shape)
 
        sess,last_saved_step = self.restore_model(model_path)
        with sess:
            summary_writer = tf.summary.FileWriter(log_path,graph=sess.graph)
            start = time.time()-0.000001 # to avoid division by 0
            for step in range(last_saved_step,training_steps):
                now   = time.time()
                speed = (step if step>0 else 1)/(now-start)
                print("{:s}: step is {:d}, speed is {:.4f} step/sec, ETA is {:s}".format(
                    time.strftime("%D %T"),
                    step,
                    speed,
                    time.strftime("%D %T",time.localtime(now+((training_steps-step)/speed)))))

                images,labels = self.data.get_batch(step,self.get_batch_size())

                # Get the z
                batch_z = np.random.uniform(-1,1,size=self.noise.shape).astype(np.float32)

                for i in range(0,int(max(1,1/generator_advantage))):
                    _, summary_str = sess.run([discriminator_optimizer,discriminator_summary],feed_dict={
                        self.images: images, 
                        self.noise : batch_z, 
                        self.labels: labels})
                    summary_writer.add_summary(summary_str,step)

                for i in range(0,int(max(1,generator_advantage))):
                    _, summary_str = sess.run([generator_optimizer,generator_summary],feed_dict={
                        self.noise : batch_z, 
                        self.labels: labels})
                    summary_writer.add_summary(summary_str,step)

                if (step%save_frequency)==0:
                    if show_count_loss:
                        discriminator_loss = sess.run(self.discriminator_loss, feed_dict={self.images: images, self.noise: batch_z, self.labels: labels})
                        generator_loss     = sess.run(self.generator_loss    , feed_dict={self.noise: batch_z, self.labels: labels})
                        print("Step {:4d}: D: loss={:.7f} G: loss={:.7f}".format(step,discriminator_loss,generator_loss))
                    save_image("{:s}/train_{:04d}.png".format(sample_path,step),self.generate_images(sess,samples_input_noise,samples_input_labels))
                    print("Step {:4d}: model is saved in {:s}".format(step,self.saver.save(sess,model_path+"/"+self.data.name,global_step=step)))
                    last_saved_step = step

            if step>last_saved_step:
                save_path = self.saver.save(sess,model_path+"/"+self.data.name,global_step=training_steps)
                print("Model saved in file: {:s}".format(save_path))

    def test(self,model_path,sample_path,input_noise=None,input_labels=None,count=0):
        sess,last_saved_step = self.restore_model(model_path)
        with sess:
            print("A sample is saved in {:s}".format(save_image(get_unique_filename(sample_path),self.generate_images(sess,input_noise,input_labels,count))))

    def get_batch_size(self):
        return int(self.images.shape[0])

    def restore_model(self,model_path):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        if latest_checkpoint is None:
            return sess,0
        self.saver.restore(sess,latest_checkpoint)
        try:
            return sess,int(re.sub("^"+model_path+"/model-([0-9]+)$","\\1",latest_checkpoint))
        except ValueError:
            return sess,0

    def generate_images(self,sess,input_noise=None,input_labels=None,count=0):
        if (input_noise is None) or (input_labels is None):
            np.random.seed()
        batch_size = self.get_batch_size()
        # if count is not specified, choose the maximum of provided noise and labels
        if count==0:
            count = max([0 if input_noise is None else input_noise.shape[0],0 if input_labels is None else input_labels.shape[0]])
            if count==0:
                count = batch_size
                
        def get_input_batch( input, start, end ):
            if len(input)<end:
                return np.append(input[start:],np.zeros((end-len(input),input.shape[1]),dtype=input.dtype),axis=0)
            return input[start:end]
        
        result = None
        for i in range(0,count,batch_size):
            if input_noise is None:
                noise = np.random.uniform(1,-1,size=self.noise.shape)
            else:
                noise = get_input_batch(input_noise,i,i+batch_size)
            if input_labels is None:
                labels = self.data.get_random_labels(self.labels.shape)
            else:
                labels = get_input_batch(input_labels,i,i+batch_size)
            batch  = sess.run(self.fakes_generator[0],feed_dict={self.noise:noise,self.labels:labels})
            result = batch if result is None else np.concatenate((result,batch))
            
        return reshape_to_square(result[0:count])

    # TODO: understand how this all works, see http://bamos.github.io/2016/08/09/deep-completion/
    def get_generator_net(self,name):
        batch_size = self.get_batch_size()
        with tf.variable_scope('generator') as scope:

            z  = tf.concat([self.noise,self.labels], 1)

            d1 = tf.nn.relu(batch_normal(fully_connect(z,output_size=1024,scope='gen_fully'),scope='gen_bn1'))
            d1 = tf.concat([d1,self.labels],1)

            # Wonder what this will be doing if the size is not divisible by 4
            h,w = self.data.shape[0]//4,self.data.shape[1]//4
            d2 = tf.nn.relu(batch_normal(fully_connect(d1,output_size=h*w*2*batch_size,scope='gen_fully2'),scope='gen_bn2'))
            d2 = tf.reshape(d2,[batch_size,h,w,2*batch_size])

            yb = tf.reshape(self.labels,shape=[batch_size,1,1,-1])
            d2 = conv_cond_concat(d2,yb)

            h,w = self.data.shape[0]//2,self.data.shape[1]//2
            d3 = tf.nn.relu(batch_normal(de_conv(d2,output_shape=[batch_size,h,w,2*batch_size], name='gen_deconv1'), scope='gen_bn3'))
            d3 = conv_cond_concat(d3,yb)

            d4 = de_conv(d3,output_shape=self.images.shape,name='gen_deconv2',initializer=xavier_initializer())

            return tf.nn.sigmoid(d4,name=name),d4

    def get_discriminator_net(self,name,images,reuse=False):
        batch_size = self.get_batch_size()
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # concat
            yb = tf.reshape(self.labels,shape=[batch_size,1,1,-1])
            concat_data = conv_cond_concat(images,yb)

            conv1, w1 = conv2d(concat_data,output_dim=self.data.get_number_of_labels(),name='dis_conv1')
            tf.add_to_collection('weight_1',w1)

            conv1 = lrelu(conv1)
            conv1 = conv_cond_concat(conv1,yb)
            tf.add_to_collection('ac_1',conv1)

            conv2, w2 = conv2d(conv1,output_dim=batch_size,name='dis_conv2')
            tf.add_to_collection('weight_2',w2)

            conv2 = lrelu(batch_normal(conv2,scope='dis_bn1'))
            tf.add_to_collection('ac_2',conv2)

            conv2 = tf.reshape(conv2,[batch_size,-1])
            conv2 = tf.concat([conv2,self.labels],1)

            # TOOD: figure out what implications the change in "output_size" will have.
            f1 = lrelu(batch_normal(fully_connect(conv2,output_size=1024,scope='dis_fully1'),scope='dis_bn2',reuse=reuse))
            f1 = tf.concat([f1,self.labels],1)

            out = fully_connect(f1,output_size=1,scope='dis_fully2',initializer=xavier_initializer())

            return tf.nn.sigmoid(out,name=name),out
