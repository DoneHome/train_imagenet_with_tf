# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os
import random
import sys
import time

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
tf.app.flags.DEFINE_integer('batch_size', 128, 'The number of batch to train in one iteration')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'The number of epochs to run')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights')
tf.app.flags.DEFINE_float('drop_out', 0.75, '')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer('save_model_steps', 1000, 'The frequency with which the model checkpoint are save.')


def main(argv=None):
    """ Train ImageNet for a number of steps. """
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Force all input processing onto CPU in order to reserve the GPU for the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            image_batch, label_batch = DataProvider.distort_input(FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.num_reader, FLAGS.num_preprocess_thread)
        # Build a Graph that computes the logits predictions from the alextnet model
        logits, _ = AlexNet.train(
            x = image_batch, keep_prob = FLAGS.drop_out,
            weight_decay = FLAGS.weight_decay)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=label_batch)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
            tf.add_to_collection('losses', cross_entropy_mean)
            # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # Variables that affect learning rate
        num_batches_per_epoch = DataProvider.TRAIN_DATASET_SIZE / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        # TODO: add polynomial, fixed, etc.
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                global_step,
                decay_steps,
                FLAGS.learning_rate_decay_factor,
                staircase=True)
        tf.summary.scalar('learning_rate', lr)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=FLAGS.momentum).minimize(loss=total_loss, global_step=global_step)

        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_batch, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        merged_summary = tf.summary.merge_all()

        coord = tf.train.Coordinator()
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

            max_steps = FLAGS.num_epochs * num_batches_per_epoch

            for step in xrange(max_steps):
                start_time = time.time()
                acc, total_loss = sess.run([accuracy, total_loss])
                duration = time.time() - start_time

                if step % FLAGS.log_every_n_steps:
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = duration
                    epoch = step / num_batches_per_epoch + 1
                    format_str = ('%s: Epoch %d  Step %d,  Loss = %.2f  Training_accuracy: %.7f  (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), epoch, step, total_loss, acc, examples_per_sec, sec_per_batch))

                if step % FLAGS.save_model_steps:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


            example, l = sess.run([image_batch, label_batch])
            summary_str = sess.run(merged_summary)

            summary_writer.add_summary(summary_str, 1)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
