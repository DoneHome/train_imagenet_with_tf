# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Converts ImageNet data to TFRecords of TF-Example protos.

This module reads the files that make up the ImageNet data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about some minutes to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

import data_utils 

FLAGS = tf.app.flags.FLAGS

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = {'train':100, 'validation':10}


tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_string('train_label_dir', None, '')
tf.app.flags.DEFINE_string('val_label_dir', None, '')

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                        feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames(dataset_dir, split_name):
    """Returns a list of filenames.

    Args:
        dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
        A list of image file paths, relative to `dataset_dir`.
    """
    dataset_dir += "/%s" %split_name
    data_root = os.path.join(dataset_dir, '')
    directories = []
    photo_filenames = []

    for filename in os.listdir(data_root):
        path = os.path.join(data_root, filename)
        if os.path.isdir(path):
            directories.append(path)
        if split_name == "validation":
            photo_filenames.append(path)

    if split_name == "train":
        for directory in directories:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                photo_filenames.append(path)

    return photo_filenames

def _map_class_to_ids(label_dir):
    name_to_label = dict()

    with open(label_dir, 'r') as rf:
        for line in rf:
            items = line.strip().split()
            sub_path, label = items
            class_name = sub_path.split("/")[0]
            name_to_label[class_name] = int(label)

    return name_to_label


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'ILSVRC_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS[split_name])
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, val_names_to_ids, dataset_dir):
    """Converts the given filenames to a TFRecord dataset.
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS[split_name])))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS[split_name]):
                output_filename = _get_dataset_filename(
                        dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        print(filenames[i])
                        image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                        height, width = image_reader.read_image_dims(sess, image_data)

                        if split_name == "train":
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            class_id = class_names_to_ids[class_name]
                        elif split_name == "validation":
                            class_name = filenames[i].split("/")[-1]
                            class_id = val_names_to_ids[class_name]
                        #print(filenames[i],class_id)

                        example = data_utils.image_to_tfexample(
                            image_data, 'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS[split_name]):
            output_filename = _get_dataset_filename(
                  dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def main(_):
    """Runs the conversion operation.

    Args:
        dataset_dir: The dataset directory where the dataset is stored.
    """
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset name with --dataset_dir')
    if not FLAGS.train_label_dir:
        raise ValueError('You must supply the dataset name with --train_label_dir')
    if not FLAGS.val_label_dir:
        raise ValueError('You must supply the dataset name with --val_label_dir')

    if not tf.gfile.Exists(FLAGS.dataset_dir):
        tf.gfile.MakeDirs(FLAGS.dataset_dir)

    if _dataset_exists(FLAGS.dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    class_names_to_ids = _map_class_to_ids(FLAGS.train_label_dir)
    val_names_to_ids = _map_class_to_ids(FLAGS.val_label_dir)

    for split_name in ['train', 'validation']:
        photo_filenames = _get_filenames(FLAGS.dataset_dir, split_name)
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        _convert_dataset(split_name, photo_filenames, class_names_to_ids, val_names_to_ids, FLAGS.dataset_dir)

if __name__ == '__main__':
    tf.app.run()
