# coding: utf-8

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('num_reader', '16', 'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer('num_gpus', 4, 'The number of gpus used for training')
tf.app.flags.DEFINE_integer('num_epochs', 4, 'The number of epochs to run')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
tf.app.flags.DEFINE_integer("start_gpu_device", 0, 'The start gpu device index')
tf.app.flags.DEFINE_integer('eval_gpu_device', 0, 'The gpu device used for eval')


def main(argv=None):
    """ Train ImageNet for a number of steps. """
    if not FLAGS.dataset.dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            images, labels = DataProvider()



if __name__ == '__main__':
    tf.app.run()
