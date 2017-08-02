# coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

IMAGE_SIZE=224

def get_files_name(dataset_dir):
    """
    """
    file_names = []

    for f in os.listdir(dataset_dir):
        if 'train' in f:
            file_names.append(f)

    return file_names

def parse_proto(serialized_example):
    """
    """
    key_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    features = tf.parse_single_example(serialized_example, features=key_to_features)

    image = tf.decode_raw(features['image/encoded'], tf.unit8)
    label = tf.cast(features['image/class/label', tf.int32])

    return image, label

def distort_input(dataset_dir, batch_size, num_reader):
    """
    """
    file_names = get_files_name(dataset_dir)

    reader = tf.TFRecordReader()

    # create a queue that produces the filenames to read
    file_queue = tf.train.string_input_producer(file_names)

    _, serialized_example = reader.read(file_queue)
    image, label = parse_proto(serialized_example)

    return image, label

