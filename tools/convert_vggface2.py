# coding: utf-8
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import glob


def _bytes_features(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert(source_dir, target_dir, crop_size, out_size, exts=[''], num_shards=128, tfrecords_prefix=''):
    if not tf.gfile.Exists(source_dir):
        print('source_dir does not exists')
        return
    if tfrecords_prefix and not tfrecords_prefix.endswith('-'):
        tfrecords_prefix += '-'

    if tf.gfile.Exists(target_dir):
        print("{} is Already exists".format(target_dir))
        return
    else:
        tf.gfile.MakeDirs(target_dir)

    # get meta-data
    path_list = []
    label_list = []
    image_classes = os.listdir(source_dir)
    for index, image_class in enumerate(image_classes):
        class_path = os.path.join(source_dir, image_class)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            image_path = os.path.join(class_path, image_name)
            path_list.append(image_path)
            label_list.append(index)
    combine = list(zip(path_list, label_list))
    np.random.shuffle(combine)
    path_list[:], label_list[:] = zip(*combine)
    combine=[]
    # shuffle path_list
    # np.random.shuffle(path_list)
    num_files = len(path_list)
    num_per_shard = num_files // num_shards # Last shard will have more files

    print('# of files: {}'.format(num_files))
    print('# of shards: {}'.format(num_shards))
    print('# files per shards: {}'.format(num_per_shard))

    # convert to tfrecords
    shard_idx = 0
    writer = None
    for i, path in enumerate(path_list):
        if i % num_per_shard == 0 and shard_idx < num_shards:
            shard_idx += 1
            tfrecord_fn = '{}{:0>4d}-of-{:0>4d}.tfrecord'.format(tfrecords_prefix, shard_idx, num_shards)
            tfrecord_path = os.path.join(target_dir, tfrecord_fn)
            print("Writing {} ...".format(tfrecord_path))
            if shard_idx > 1:
                writer.close()
            writer = tf.python_io.TFRecordWriter(tfrecord_path)

        # mode='RGB' read even grayscale image as RGB shape
        # im_HR = scipy.misc.imread(path, mode='RGB')
        im_HR = tf.gfile.FastGFile(path, 'rb').read()  # 读入图片
        index = label_list[i]
        example = tf.train.Example(features=tf.train.Features(feature={
            # "image_LR": _bytes_features([im_LR.tostring()]),
            # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'label': _int64_features(index),
            "image_raw": _bytes_features(im_HR)
        }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    # CelebA
    convert('/media/lab225/Document2/merle/faceDataset/vggface2_align_112x96',
            '/media/lab225/Document2/merle/faceDataset/vggface2_align_112x96_tfrecord',
            crop_size=[112, 112], out_size=[112, 112], exts=['jpg'], num_shards=64, tfrecords_prefix='VGGFace2')
