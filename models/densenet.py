from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, kernel_size=[kernel, kernel], stride=stride,
                            padding='SAME', data_format='NHWC', activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, kernel_size=[kernel, kernel], stride=stride,
                            padding='SAME', data_format='NHWC', activation_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)

# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg

def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output

# The dense layer
def denseConvlayer(layer_inputs, bottleneck_scale, growth_rate, is_training):
    # Build the bottleneck operation
    net = layer_inputs
    net_temp = tf.identity(net)
    net = batchnorm(net, is_training)
    net = prelu_tf(net, name='Prelu_1')
    net = conv2(net, kernel=1, output_channel=bottleneck_scale*growth_rate, stride=1, use_bias=False, scope='conv1x1')
    net = batchnorm(net, is_training)
    net = prelu_tf(net, name='Prelu_2')
    net = conv2(net, kernel=3, output_channel=growth_rate, stride=1, use_bias=False, scope='conv3x3')

    # Concatenate the processed feature to the feature
    net = tf.concat([net_temp, net], axis=3)

    return net


# The transition layer
def transitionLayer(layer_inputs, output_channel, is_training):
    net = layer_inputs
    net = batchnorm(net, is_training)
    net = prelu_tf(net)
    net = conv2(net, 1, output_channel, stride=1, use_bias=False, scope='conv1x1')

    return net


# The dense block
def denseBlock(block_inputs, num_layers, bottleneck_scale, growth_rate, is_training):
    # Build each layer consecutively
    net = block_inputs
    for i in range(num_layers):
        with tf.variable_scope('dense_conv_layer%d'%(i+1)):
            net = denseConvlayer(net, bottleneck_scale, growth_rate, is_training)

    return net


# Here we define the dense block version generator
def densenet(gen_inputs, gen_output_channels, reuse=False):
    # The main netowrk
    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input stage
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        # The dense block part
        # Define the denseblock configuration
        layer_per_block = 16
        bottleneck_scale = 4
        growth_rate = 12
        transition_output_channel = 128
        with tf.variable_scope('denseBlock_1'):
            net = denseBlock(net, layer_per_block, bottleneck_scale, growth_rate, FLAGS)

        with tf.variable_scope('transition_layer_1'):
            net = transitionLayer(net, transition_output_channel, FLAGS.is_training)

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

        return net

def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=512, weight_decay=0.0, reuse=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        return densenet(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)

def densenet(inputs, is_training=True, dropout_keep_prob=0.4,
                        bottleneck_layer_size=512, reuse=None,
                        scope='densenet'):
    end_points = {}
    with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):

                # 55 x 47 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                print('Conv2d_1a_3x3:', net.get_shape().as_list())
                # 53 x 45 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID', scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                print('Conv2d_2a_3x3:', net.get_shape().as_list())
                # 53 x 45 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                print('Conv2d_2b_3x3:', net.get_shape().as_list())
                # 26 x 22 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                print('MaxPool_3a_3x3:', net.get_shape().as_list())
                # The dense block part
                # Define the denseblock configuration
                layer_per_block = 8
                bottleneck_scale = 4
                growth_rate = 12
                transition_output_channel = 128
                with tf.variable_scope('denseBlock_1'):
                    net = denseBlock(net, layer_per_block, bottleneck_scale, growth_rate, is_training)
                end_points['denseBlock_1'] = net
                print('denseBlock_1:', net.get_shape().as_list())

                with tf.variable_scope('transition_layer_1'):
                    net = transitionLayer(net, transition_output_channel, is_training)
                end_points['transition_layer_1'] = net
                print('transition_layer_1:', net.get_shape().as_list())

                # 12 x 10 x 128
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_4a_3x3')
                end_points['MaxPool_4a_3x3'] = net
                print('MaxPool_4a_3x3:', net.get_shape().as_list())
                # The dense block part
                # Define the denseblock configuration
                layer_per_block = 16
                bottleneck_scale = 4
                growth_rate = 12
                transition_output_channel = 256
                with tf.variable_scope('denseBlock_2'):
                    net = denseBlock(net, layer_per_block, bottleneck_scale, growth_rate, is_training)
                end_points['denseBlock_2'] = net
                print('denseBlock_2:', net.get_shape().as_list())

                with tf.variable_scope('transition_layer_2'):
                    net = transitionLayer(net, transition_output_channel, is_training)
                end_points['transition_layer_2'] = net
                print('transition_layer_2:', net.get_shape().as_list())

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool')
                    net = slim.flatten(net)
                    print('AvgPool:', net.get_shape().as_list())
                    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                    #                    scope='Dropout')
                    # end_points['PreLogitsFlatten'] = net
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                        scope='Bottleneck', reuse=False)
    return net, end_points




