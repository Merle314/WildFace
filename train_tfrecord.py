# -*- coding: utf-8 -*-
import argparse
import glob
import importlib
import math
import os
import os.path
import random
import sys
import time
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy import misc
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, data_flow_ops

import facenet
import lfw
# from AM_softmax import AM_logits_compute
from loss_func import adaptive_loss, cosSoftmax_loss
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def main(args):
    network = importlib.import_module(args.model_def)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    np.random.seed(seed=args.seed)
    random.seed(args.seed)        
    
    nrof_classes = 8631
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
        # images_lfw = np.load(args.lfw_dir+'.npy')
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        tfrecords_list = glob.glob(args.tfrecord_dir)
        filename_queue = tf.train.string_input_producer(tfrecords_list, shuffle=True)
        reader = tf.TFRecordReader()
        key, records = reader.read(filename_queue)
        features = tf.parse_single_example(records,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                "image_raw": tf.FixedLenFeature([], tf.string)
            }
        )
        label = tf.cast(features['label'], tf.int32)
        image = tf.cast(tf.image.decode_jpeg(features["image_raw"], channels=3), tf.float32)
        if args.random_flip:
            image = tf.image.random_flip_left_right(image)
        image.set_shape((112, 96, 3))
        image = tf.subtract(image,127.5) * 0.0078125
        images_and_labels = [image, label]
        image_batch, label_batch = tf.train.shuffle_batch(
            images_and_labels, batch_size=args.batch_size,
            shapes=[(112, 96, 3), ()],
            capacity=40000, min_after_dequeue=10000,
            allow_smaller_final_batch=True)
        # image_batch.set_shape([args.batch_size, 112, 112, 3])
        # label_batch.set_shape([args.batch_size])
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label')
        image_batch.set_shape([None, 112, 96, 3])

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: 3000000')
        print('Building training graph')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        prelogits = tf.identity(prelogits, 'prelogits')
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        cross_entropy_mean, logits = cosSoftmax_loss(embeddings, label_batch, args.batch_size, nrof_classes, m=0.35, s=30, name='softmax')
        # cross_entropy_mean, accuracy = adaptive_loss(embeddings, label_batch, args.batch_size, nrof_classes, m=0.35, s=30, name='softmax')
        # AM_logits, logits = AM_logits_compute(embeddings, label_batch, args, nrof_classes)
        #AM_logits = Arc_logits(embeddings, label_batch, args, nrof_classes)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=label_batch, logits=AM_logits, name='cross_entropy_per_example')
        # cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
       
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        for weights in slim.get_variables_by_name('kernel'):
            kernel_regularization = tf.contrib.layers.l2_regularizer(args.weight_decay)(weights)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, kernel_regularization)	
	
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        if args.weight_decay==0:
            total_loss = tf.add_n([cross_entropy_mean], name='total_loss')
        else:
            total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
        tf.add_to_collection('losses', total_loss)

        #define two saver in case under 'finetuning on different dataset' situation 
        saver_save = tf.train.Saver(tf.trainable_variables(), max_to_keep =1)

        train_op = facenet.train(total_loss, global_step, args.optimizer, 
           learning_rate, args.moving_average_decay, tf.trainable_variables(), args.log_histograms)
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step = global_step,var_list=tf.trainable_variables())
        # train_op = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(total_loss,global_step=global_step,var_list=tf.trainable_variables())
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        with sess.as_default():
            if pretrained_model:
                exclusions = []
                except_exclusions = slim.get_variables_to_restore(exclude=exclusions)
                restore_variables = [v for v in tf.trainable_variables() if v in except_exclusions]
                saver_load= tf.train.Saver(restore_variables)
                print('Restoring pretrained model: %s' % pretrained_model)
                saver_load.restore(sess, pretrained_model)

            best_accuracy = evaluate_double(sess, phase_train_placeholder, image_batch,
                embeddings, lfw_paths, actual_issame, log_dir, 0, summary_writer,
                0.0, saver_save, model_dir, subdir, args)

            print('Running training')
            epoch = 0
            best_accuracy = 0.0

            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                train(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder,
                      global_step, total_loss, train_op, summary_op, summary_writer,
                      regularization_losses, accuracy, learning_rate)

                print('validation running...')
                if args.lfw_dir:
                    best_accuracy = evaluate_double(sess, phase_train_placeholder, image_batch,
                        embeddings, lfw_paths, actual_issame, log_dir, step+args.epoch_size, summary_writer,
                        best_accuracy, saver_save, model_dir, subdir, args)
    return model_dir

def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold

def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset

def train(args, sess, epoch, learning_rate_placeholder, phase_train_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, accuracy, learning_rate):
    batch_number = 0
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(args.learning_rate_schedule_file, epoch)

    print('training a epoch...')
    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, accuracy_, lr_, summary_str = sess.run(
                [loss, train_op, global_step, regularization_losses, accuracy, learning_rate, summary_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss, accuracy_, lr_ = sess.run(
                [loss, train_op, global_step, regularization_losses, accuracy, learning_rate],
                feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch:[%d][%d/%d]\tTime:%.2f  Loss:%2.3f  Lr:%2.3f Acc:%2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, err, lr_, accuracy_))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step

def load_data(image_paths):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, 112, 96, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        img = (img*1.0-127.5)/128
        images[i,:,:,:] = img
    return images

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def evaluate_with_no_cv(emb_array, actual_issame):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = emb_array[0::2]
    embeddings2 = emb_array[1::2]

    nrof_thresholds = len(thresholds)
    accuracys = np.zeros((nrof_thresholds))
    
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff),1)
    # dist = np.arccos(np.sum(embeddings1*embeddings2, axis=1))/np.pi

    for threshold_idx, threshold in enumerate(thresholds):
        _, _, accuracys[threshold_idx] = facenet.calculate_accuracy(threshold, dist, actual_issame)

    best_acc = np.max(accuracys)
    best_thre = thresholds[np.argmax(accuracys)]
    return best_acc,best_thre

def evaluate_double(sess, phase_train_placeholder, image_batch, embeddings, image_paths,
    actual_issame, log_dir, step, summary_writer, best_accuracy, saver_save, model_dir,
    subdir, args):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    images_lfw = np.load(args.lfw_dir+'.npy')
    images_lfw = images_lfw.reshape((-1, 112, 96, 3))
    batch_size = args.lfw_batch_size
    nrof_images = images_lfw.shape[0]
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    #math.ceil为向上取整，意味这最后一个batch可能样本数少于batch_size
    emb_array = np.zeros((nrof_images, args.embedding_size))
    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)#保证最后一个batch的正确性
        images = images_lfw[start_index:end_index]
        #by charles
        images_flip = np.flip(images, 2)
        feed_dict = { image_batch:images, phase_train_placeholder:False }
        feed_dict_flip = { image_batch:images_flip, phase_train_placeholder:False }
        emb = sess.run(embeddings, feed_dict=feed_dict)
        emb_flip = sess.run(embeddings, feed_dict=feed_dict_flip)
        emb_average = (emb + emb_flip)/2.0
        emb_array[start_index:end_index,:] = emb_average
    accuracy,thre = evaluate_with_no_cv(emb_array, actual_issame)
    if np.mean(accuracy) > best_accuracy:
        save_variables_and_metagraph(sess, saver_save, summary_writer, model_dir, subdir, step)
        best_accuracy = np.mean(accuracy)
    print('Accuracy: %1.3f Threshold: %1.3f' % (accuracy,thre))
    with open(os.path.join(log_dir, "lfw_accuracy.txt"),"a") as f:
        f.write('step:%d\tAccuracy:%1.4f  Threshold:%1.4f\n' % (step, accuracy, thre))

    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=accuracy)
    summary.value.add(tag='lfw/threshold', simple_value=thre)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    images_lfw = []
    return best_accuracy


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, help='Directory where to write event logs.',
        default='/data/Merle/Models/WildFace')
    parser.add_argument('--models_base_dir', type=str, help='Directory where to write trained models and checkpoints.',
        default='/data/Merle/Models/WildFace')
    parser.add_argument('--gpu_memory_fraction', type=float, help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=1.0)
    parser.add_argument('--pretrained_model', type=str, help='Load a pretrainedmodel before training starts.')
        # default='/data/Merle/Models/WildFace/20181115-041701/model-20181115-041701.ckpt-21000')
        # default='/media/lab225/Documents/merle/faceDataSet/Models/20181007-144210/model-20181007-144210.ckpt-79000')
        # default='/media/lab225/Documents/merle/faceDataSet/Models/20181027-170254/model-20181027-170254.ckpt-29000')
        # default='/media/lab225/Documents/merle/faceDataSet/Models/20180402-114759/model-20180402-114759.ckpt-275')
        # default='/media/lab225/Documents/merle/faceDataSet/Models/20180904-220157/model-20180904-220157.ckpt-40000')
        # default='/media/lab225/Documents/merle/faceDataSet/Models/20180910-101626/model-20180910-101626.ckpt-11000')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='/data/Merle/Dataset/vggface2_align_112x96')
    parser.add_argument('--tfrecord_dir', type=str, help='Path to the data directory containing aligned faces converted to tfrecord.',
        default='/data/Merle/Dataset/vggface2_align_112x96_tfrecord/*.tfrecord')
        # default='/media/lab225/Document2/merle/faceDataset/ms1m_tfrecords/*.tfrecords')
    parser.add_argument('--model_def', type=str, help='Model definition. Points to a module containing the definition of the inference graph.',
        default='models.resface_mul')
    parser.add_argument('--max_nrof_epochs', type=int, help='Number of epochs to run.',
        default=1000)
    # parser.add_argument('--image_size', type=int, help='Number of epochs to run.',
    #     default=(112, 96, 3))
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.',
        default=32)
    parser.add_argument('--epoch_size', type=int, help='Number of batches per epoch.',
        default=1000)
    parser.add_argument('--embedding_size', type=int, help='Dimensionality of the embedding.',
        default=512)
    parser.add_argument('--random_flip', help='Performs random horizontal flipping of training images.',
        default=True)
    parser.add_argument('--keep_probability', type=float, help='Keep probability of dropout for the fully connected layer(s).',
        default=0.4)
    parser.add_argument('--weight_decay', type=float, help='L2 weight regularization.',
        default=5e-4)
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate. If set to a negative value a learning rate schedule can be specified in the file "learning_rate_schedule.txt"',
        default=-1)
    parser.add_argument('--learning_rate_decay_epochs', type=int, help='Number of epochs between learning rate decay.',
        default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float, help='Learning rate decay factor.',
        default=1.0)
    parser.add_argument('--seed', type=int, help='Random seed.',
        default=666)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'], help='The optimization algorithm to use',
        default='ADAM')
    parser.add_argument('--moving_average_decay', type=float, help='Exponential decay for tracking of training parameters.',
        default=0.9999)
    parser.add_argument('--log_histograms', help='Enables logging of weight/bias histograms in tensorboard.',
        default=False)
    parser.add_argument('--nrof_preprocess_threads', type=int, help='Number of preprocessing (data loading and augmentation) threads.',
        default=4)
    parser.add_argument('--learning_rate_schedule_file', type=str, help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
        default='data/learning_rate_AM_softmax.txt')
    parser.add_argument('--filter_filename', type=str, help='File containing image data used for dataset filtering',
        default='')
    parser.add_argument('--filter_percentile', type=float, help='Keep only the percentile images closed to its class center',
        default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int, help='Keep only the classes with this number of examples or more',
        default=0)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str, help='The file containing the pairs to use for validation.',
        default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str, help='The file extension for the LFW dataset.',
        default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str, help='Path to the data directory containing aligned face patches.',
        default='/data/Merle/Dataset/LFW/align_112x96')
    parser.add_argument('--lfw_batch_size', type=int, help='Number of images to process in a batch in the LFW test set.',
        default=100)
    parser.add_argument('--lfw_nrof_folds', type=int, help='Number of folds to use for cross validation. Mainly used for testing.',
        default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
