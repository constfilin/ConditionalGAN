import tensorflow as tf

# The implementations of leakyRelu
def lrelu(x , alpha=0.2, name="lrelu"):
    return tf.maximum(x,alpha*x,name=name)

def conv2d(input_,output_dim,k_h,k_w,d_h,d_w,
           scope              = "conv2d",
           weights_initializer= tf.truncated_normal_initializer(stddev=0.02),
           biases_initializer = tf.constant_initializer(0.0)):
    with tf.variable_scope(scope):
        weights = tf.get_variable('w',[k_h, k_w, input_.get_shape()[-1],output_dim],initializer=weights_initializer)
        conv    = tf.nn.conv2d(input_,weights,strides=[1,d_h,d_w,1],padding='SAME')
        biases  = tf.get_variable('biases', [output_dim], initializer=biases_initializer)
        conv    = tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape())
        return conv,weights

def deconv2d(input_,output_shape,k_h,k_w,d_h,d_w,
             scope              = "deconv2d",
             weights_initializer= tf.truncated_normal_initializer(stddev=0.02),
             biases_initializer = tf.constant_initializer(0.0)):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        weights = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],initializer=weights_initializer)
        deconv  = tf.nn.conv2d_transpose(input_, weights, output_shape=output_shape,strides=[1, d_h, d_w, 1])
        biases  = tf.get_variable('biases', [output_shape[-1]], initializer=biases_initializer)
        return tf.reshape(tf.nn.bias_add(deconv,biases),deconv.get_shape())

def linear(input_,output_size,
           scope              = "linear",
           weights_initializer= tf.truncated_normal_initializer(stddev=0.02),
           biases_initializer = tf.constant_initializer(0.0)):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,initializer=weights_initializer)
        biases = tf.get_variable("bias",   [output_size], initializer=biases_initializer)
        return tf.nn.bias_add(tf.matmul(input_,matrix),biases)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x,y*tf.ones([x_shapes[0],x_shapes[1],x_shapes[2],y_shapes[3]])],3)

def batch_norm(input_, is_training, scope="scope", reuse=False):
    return tf.contrib.layers.batch_norm(
        input_,
        decay=0.999,
        epsilon=0.001,
        updates_collections=None,
        scale=True,
        fused=False,
        is_training=is_training,
        scope=scope,
        reuse=reuse)
