import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
Resface20 and Resface36 proposed in sphereface and applied in Additive Margin Softmax paper
Notice:
batch norm is used in line 111. to cancel batch norm, simply commend out line 111 and use line 112
'''

def prelu(x):
    with tf.variable_scope('PRelu'):   
        alphas = tf.Variable(tf.constant(0.25,dtype=tf.float32,shape=[x.get_shape()[-1]]),name='prelu_alphas')
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg

# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = inputs.get_shape().as_list()
    batch_size = -1
    h = size[1]
    w = size[2]
    c = size[-1]

    shape_1 = [batch_size, h//scale, scale, w//scale, scale]
    shape_2 = [batch_size, h//scale, w//scale, 4]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, c, axis=3)
    input_1 = []
    input_2 = []
    input_3 = []
    input_4 = []
    for x in input_split:
        x_1, x_2, x_3, x_4 = phaseShift(x, shape_1, shape_2, scale=scale)
        input_1.append(x_1)
        input_2.append(x_2)
        input_3.append(x_3)
        input_4.append(x_4)
    return(tf.concat(input_1, axis=3), tf.concat(input_2, axis=3), tf.concat(input_3, axis=3), tf.concat(input_4, axis=3))

def phaseShift(inputs, shape_1, shape_2, scale=2):
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])
    return tf.split(tf.reshape(X, shape_2), scale*scale, axis=3)

def resface_block(lower_input,output_channels,scope=None):
    with tf.variable_scope(scope):
        net = slim.conv2d(lower_input, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = slim.conv2d(net, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return lower_input + net

# Define the convolution block before the resface layer
def resface_pre(lower_input,output_channels,scope=None):
    net_1, net_2, net_3, net_4 = pixelShuffler(lower_input, scale=2)
    # net = slim.conv2d(net_1, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # net = slim.conv2d(net_2+net, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # net = slim.conv2d(net_3+net, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    # net = slim.conv2d(net_4+net, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    flatten = slim.flatten(lower_input)
    alpha_1 = tf.expand_dims(tf.expand_dims(slim.fully_connected(flatten, 1, activation_fn=tf.nn.sigmoid, scope='alpha_1', reuse=False), -1), -1)
    alpha_2 = tf.expand_dims(tf.expand_dims(slim.fully_connected(flatten, 1, activation_fn=tf.nn.sigmoid, scope='alpha_2', reuse=False), -1), -1)
    alpha_3 = tf.expand_dims(tf.expand_dims(slim.fully_connected(flatten, 1, activation_fn=tf.nn.sigmoid, scope='alpha_3', reuse=False), -1), -1)
    alpha_4 = tf.expand_dims(tf.expand_dims(slim.fully_connected(flatten, 1, activation_fn=tf.nn.sigmoid, scope='alpha_4', reuse=False), -1), -1)
    net_1 = slim.conv2d(net_1, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net_2 = slim.conv2d(net_2, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net_3 = slim.conv2d(net_3, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net_4 = slim.conv2d(net_4, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return net_1*alpha_1+net_2*alpha_2+net_3*alpha_3+net_4*alpha_4
    # return net_1+net_2+net_3+net_4

# # Define the convolution block before the resface layer
# def resface_pre(lower_input,output_channels,scope=None):
#     net = slim.conv2d(lower_input, output_channels, stride=2, scope=scope)
#     return net

def resface20(images, keep_probability,
              phase_train=True, bottleneck_layer_size=512,
              weight_decay=0.0, reuse=None):
    '''
    conv name
    conv[conv_layer]_[block_index]_[block_layer_index]
    '''
    end_points = {}
    with tf.variable_scope('Conv1'):
        print('input:', images.get_shape().as_list())
        # net = slim.conv2d(images, 64, scope='input_pre')
        # print('input_pre:', net.get_shape().as_list())
        net = resface_pre(images,64,scope='Conv1_pre')
        end_points['Conv1_pre'] = net
        print('Conv1_pre:', net.get_shape().as_list())
        net = slim.repeat(net,1,resface_block,64,scope='Conv1')
        end_points['Conv1'] = net
        print('Conv1:', net.get_shape().as_list())
    with tf.variable_scope('Conv2'):
        # net = slim.conv2d(net, 128, scope='input_pre')
        net = resface_pre(net,128,scope='Conv2_pre')
        end_points['Conv2_pre'] = net
        print('Conv2_pre:', net.get_shape().as_list())
        net = slim.repeat(net,2,resface_block,128,scope='Conv2')
        end_points['Conv2'] = net
        print('Conv2:', net.get_shape().as_list())
    with tf.variable_scope('Conv3'):
        # net = slim.conv2d(net, 256, scope='input_pre')
        net = resface_pre(net,256,scope='Conv3_pre')
        end_points['Conv3_pre'] = net
        print('Conv3_pre:', net.get_shape().as_list())
        net = slim.repeat(net,4,resface_block,256,scope='Conv3')
        end_points['Conv3'] = net
        print('Conv3:', net.get_shape().as_list())
    with tf.variable_scope('Conv4'):
        # net = slim.conv2d(net, 512, scope='input_pre')
        net = resface_pre(net,512,scope='Conv4_pre')
        end_points['Conv4_pre'] = net
        print('Conv4_pre:', net.get_shape().as_list())
        net = slim.repeat(net,1,resface_block,512,scope='Conv4')
        end_points['Conv4'] = net
        print('Conv4:', net.get_shape().as_list())
    with tf.variable_scope('Logits'):
        #pylint: disable=no-member
        # net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
        #                      scope='AvgPool')
        net = slim.flatten(net)
        end_points['flatten'] = net
        print('flatten:', net.get_shape().as_list())
        flatten = slim.dropout(net, keep_probability, is_training=phase_train,
                           scope='Dropout')
        end_points['Dropout'] = net
        print('Dropout:', net.get_shape().as_list())
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
            scope='Bottleneck', reuse=False)
    end_points['Bottleneck'] = net
    print('Bottleneck', net.get_shape().as_list())
    return net, end_points

def resface36(images, keep_probability, 
             phase_train=True, bottleneck_layer_size=512, 
             weight_decay=0.0, reuse=None):
    '''
    conv name
    conv[conv_layer]_[block_index]_[block_layer_index]
    '''
    with tf.variable_scope('Conv1'):
        net = resface_pre(images,64,scope='Conv1_pre')
        net = slim.repeat(net,2,resface_block,64,scope='Conv_1')
    with tf.variable_scope('Conv2'):
        net = resface_pre(net,128,scope='Conv2_pre')
        net = slim.repeat(net,4,resface_block,128,scope='Conv_2')
    with tf.variable_scope('Conv3'):
        net = resface_pre(net,256,scope='Conv3_pre')
        net = slim.repeat(net,8,resface_block,256,scope='Conv_3')
    with tf.variable_scope('Conv4'):
        net = resface_pre(net,512,scope='Conv4_pre')
        #net = resface_block(Conv4_pre,512,scope='Conv4_1')
        net = slim.repeat(net,1,resface_block,512,scope='Conv4')

    with tf.variable_scope('Logits'):
        #pylint: disable=no-member
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                             scope='AvgPool')
        net = slim.flatten(net)
        net = slim.dropout(net, keep_probability, is_training=phase_train,
                           scope='Dropout')
    prelogits = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
            scope='Bottleneck', reuse=False)  
    return prelogits

def inference(image_batch, keep_probability, 
              phase_train=True, bottleneck_layer_size=512, 
              weight_decay=0.0):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'scale':True,
        'is_training': phase_train,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }    
    with tf.variable_scope('Resface'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             weights_regularizer=slim.l2_regularizer(weight_decay), 
                             activation_fn=prelu,
                             normalizer_fn=slim.batch_norm,
                             #normalizer_fn=None,
                             normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.conv2d], kernel_size=3):
                return resface20(images=image_batch, 
                                keep_probability=keep_probability, 
                                phase_train=phase_train, 
                                bottleneck_layer_size=bottleneck_layer_size, 
                                reuse=None)
