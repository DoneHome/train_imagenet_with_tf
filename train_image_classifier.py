# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

import dataSet as DataProvider
from alexnet import AlexNet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('train_dir', None, 'The directory where the event logs and checkpoint are stored.')
tf.app.flags.DEFINE_integer("start_gpu_device", 0, 'The start gpu device index')
tf.app.flags.DEFINE_integer('eval_gpu_device', 0, 'The gpu device used for eval')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus used for training')

tf.app.flags.DEFINE_integer('num_reader', '4', 'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer('num_preprocess_thread', '4', 'The number of threads that preprocess the image.')

tf.app.flags.DEFINE_integer('num_class', 1000, 'The number of classes for the ImageNet dataset.')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'The name of the optimizer, one of "adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop"')

tf.app.flags.DEFINE_integer('batch_size', 50, 'The number of batch to train in one iteration')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'The number of epochs to run')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights')

tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')


def main(argv=None):
    """ Train ImageNet for a number of steps. """
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    with tf.Graph().as_default():
        # Force all input processing onto CPU in order to reserve the GPU for the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            image_batch, label_batch = DataProvider.distort_input(FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.num_reader, FLAGS.num_preprocess_thread)
        # Build a Graph that computes the logits predictions from the alextnet model
        logits, _ = AlexNet.train(image_batch)
        #labels = tf.one_hot(label_batch, FLAGS.num_class)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
            tf.add_to_collection('losses', cross_entropy_mean)
            # add l2 loss


        merged_summary = tf.summary.merge_all()

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            example, l = sess.run([image_batch, label_batch])
            summary_str = sess.run(merged_summary)

            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)
            summary_writer.add_summary(summary_str, 1)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
