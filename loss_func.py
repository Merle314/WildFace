import tensorflow as tf


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def cosSoftmax_loss(features, labels, batch_size, nrof_classes, m=0.35, s=30, name='softmax'):
    with tf.variable_scope(name):
        nrof_features = features.get_shape()[1]
        kernel = tf.get_variable('kernel', [nrof_features, nrof_classes], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
        # kernel = tf.Variable(tf.truncated_normal([nrof_features, nrof_classes]))
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name=name+'/kernel_norm')
        cos_theta = tf.matmul(features, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1) # for numerical steady
        phi = cos_theta-m
        label_onehot = tf.one_hot(labels, nrof_classes)
        adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)
        loss_cosSoftmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=adjust_theta))

        kernel_sub = kernel_norm-tf.reduce_mean(kernel_norm, axis=1, keep_dims=True)
        loss_cov = tf.reduce_mean(tf.square(tf.matmul(kernel_sub, tf.transpose(kernel_sub))))

        loss = loss_cosSoftmax+loss_cov
    return loss, cos_theta

def adaptive_loss(features, labels, batch_size, nrof_classes, m=0.35, s=30, name='softmax'):
    with tf.variable_scope(name):
        nrof_features = features.get_shape()[1]
        kernel = tf.get_variable('kernel', [nrof_features, nrof_classes*2], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name=name+'/kernel_norm')
        cos_theta = tf.matmul(features, kernel_norm)
        cos_theta = tf.clip_by_value(cos_theta, -1, 1) # for numerical steady
        phi = cos_theta-m

        label_onehot = tf.one_hot(labels, nrof_classes*2)
        adjust_theta = s * tf.where(tf.equal(label_onehot,1), phi, cos_theta)
        loss_cosSoftmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=adjust_theta))

        labels_adjust = tf.argmax(cos_theta, axis=1)
        label_adjust_onehot = tf.one_hot(labels_adjust, nrof_classes*2)
        adjust_theta_adjust = s * tf.where(tf.logical_and(tf.equal(label_adjust_onehot,1), tf.equal(label_adjust_onehot,label_onehot)), phi, cos_theta)
        loss_cosSoftmax_adjust = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_adjust, logits=adjust_theta_adjust))

        correct_prediction = tf.cast(tf.equal(tf.argmax(cos_theta, 1), tf.cast(labels, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        loss = tf.cond(accuracy < 0.80, lambda: tf.identity(loss_cosSoftmax), lambda: tf.identity(loss_cosSoftmax_adjust))
        # ratio = tf.maximum(accuracy, 0.95)
        # loss = accuracy*loss_cosSoftmax_adjust + (1-accuracy)*loss_cosSoftmax
        # loss = loss_cosSoftmax
    return loss, accuracy
