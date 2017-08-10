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
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights')
tf.app.flags.DEFINE_float('drop_out', 0.75, '')
tf.app.flags.DEFINE_integer('log_every_n_steps', 10, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer('save_summaries_steps', 100, 'The frequency with which summary are save.')
tf.app.flags.DEFINE_integer('save_model_steps', 1000, 'The frequency with which the model checkpoint are save.')

# fixed parameter
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9, 'The decay to use for the moving average.')


def _configure_session():
    """
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = True

    return config

def _configure_learning_rate(num_batches_per_epoch, global_step):
    """
    """
    # Variables that affect learning rate
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    # TODO: add polynomial, fixed, etc.
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

    return learning_rate

def _configure_optimizer(learning_rate):
    """
    """
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=FLAGS.momentum)
    return optimizer


def main(argv=None):
    """ Train ImageNet for a number of steps. """
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    if not FLAGS.train_dir:
        raise ValueError('You must supply the dataset directory with --train_dir')
    if not FLAGS.num_reader:
        raise ValueError('Please make num_readers at least 1')

    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Force all input processing onto CPU in order to reserve the GPU for the forward inference and back-propagation.
        with tf.device('/cpu:0'):
            image_batch, label_batch = DataProvider.distort_input(FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.num_reader, FLAGS.num_preprocess_thread, is_train=True)
        # Build a Graph that computes the logits predictions from the alextnet model
        logits, _ = AlexNet.train(
            x = image_batch, keep_prob = FLAGS.drop_out,
            weight_decay = FLAGS.weight_decay)


        num_batches_per_epoch = int(DataProvider.TRAIN_DATASET_SIZE / FLAGS.batch_size)

        with tf.name_scope('learning_rate'):
            learning_rate = _configure_learning_rate(num_batches_per_epoch, global_step)
            tf.summary.scalar('learning_rate', learning_rate)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=label_batch)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
            tf.add_to_collection('losses', cross_entropy_mean)

        with tf.name_scope('total_loss'):
            # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        with tf.name_scope('optimizer'):
            optimizer = _configure_optimizer(learning_rate)
            grads = optimizer.compute_gradients(total_loss)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        #with tf.name_scope('accuracy'):
        #    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_batch, 1))
        #    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        #    tf.summary.scalar('accuracy', accuracy)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session(config=_configure_session()) as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

            max_steps = int(FLAGS.num_epochs * num_batches_per_epoch)

            for step in xrange(max_steps):
                start_time = time.time()
                _, loss_value = sess.run([train_op, total_loss])
                duration = time.time() - start_time

                if step % FLAGS.log_every_n_steps == 0:
                    examples_per_sec = FLAGS.batch_size / duration
                    sec_per_batch = duration
                    epoch = step / num_batches_per_epoch + 1
                    format_str = ('%s: Epoch %d  Step %d  Total_loss = %.2f  (%.1f examples/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), epoch, step, loss_value, examples_per_sec, sec_per_batch))

                if step % FLAGS.save_summaries_steps == 0:
                    # Visual Training Process
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % FLAGS.save_model_steps == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
