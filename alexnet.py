# coding: utf-8

import tensorflow as tf


def weight_init(shape, stddev, name):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32), name=name)

def bias_init(value, shape, name):
    return tf.Variable(tf.constant(value, shape=shape, dtype=tf.float32), name=name)

def conv2d(x, w, stride, padding):
    return tf.nn.conv2d(x, w, [1, stride[0], stride[1], 1], padding=padding)

def max_pool(x, size, stride, padding):
    return tf.nn.max_pool(x, [1, stride[0], stride[1], 1], [1, stride[0], stride[1], 1], padding=padding)

def lrn(x, radius, alpha, beta, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


class AlexNet(object):

    def __init__(self):
        self.keep_prob= 0.75
        pass

    def train(self, x):
        """
        """
        with tf.name_scope('conv_net') as conv_scope:
            with tf.name_scope('conv1') as scope:
                w1 = weight_init([11, 11, 3, 96], 0.01, 'w_conv1')
                b1 = bias_init(0.0, [96], 'b_conv1')
                conv1 = tf.add(conv2d(x, w1, stride=[4, 4], padding='SAME'), b1)
                relu1 = tf.nn.relu(conv1)
                norm1 = lrn(relu1, radius=2, alpha=1e-4, beta=0.75, bias=1.0)
                pool1 = max_pool(norm1, size=[3,3], stride=[2,2], padding='VALID')

            with tf.name_scope('conv2') as scope:
                w2 = weight_init([5, 5, 96, 256], 0.01, 'w_conv2')
                b2 = bias_init(0.0, [256], 'b_conv2')
                conv2 = tf.add(conv2d(pool1, w2, stride=[1, 1], padding='SAME'), b2)
                relu2 = tf.nn.relu(conv2)
                norm2 = lrn(relu2, radius=2, alpha=1e-4, beta=0.75, bias=1.0)
                pool2 = max_pool(norm2, size=[3,3], stride=[2,2], padding='VALID')

            with tf.name_scope('conv3') as scope:
                w3 = weight_init([3, 3, 256, 384], 0.01, 'w_conv3')
                b3 = bias_init(0.0, [384], 'b_conv3')
                conv3 = tf.add(conv2d(pool2, w3, stride=[1, 1], padding='SAME'), b3)
                relu3 = tf.nn.relu(conv3)

            with tf.name_scope('conv4') as scope:
                w4 = weight_init([3, 3, 384, 384], 0.01, 'w_conv4')
                b4 = bias_init(0.0, [384], 'b_conv4')
                conv4 = tf.add(conv2d(relu3, w4, stride=[1, 1], padding='SAME'), b4)
                relu4 = tf.nn.relu(conv4)

            with tf.name_scope('conv5') as scope:
                w5 = weight_init([3, 3, 384, 256], 0.01, 'w_conv4')
                b5 = bias_init(0.0, [256], 'b_conv5')
                conv5 = tf.add(conv2d(relu4, w5, stride=[1, 1], padding='SAME'), b5)
                relu5 = tf.nn.relu(conv5)
                pool5 = max_pool(relu5, size=[3,3], stride=[2,2], padding='VALID')

        with tf.name_scope('fc_layer') as fc_scope:
            with tf.name_scope('fc6') as scope:
                flatten_input = tf.reshape(pool5, [-1, 6*6*256])

                w_fc6 = weight_init(shape=[6*6*256, 4096], stddev=0.01, name='w_fc6')
                b_fc6 = bias_init(0.0, [4096], 'b_fc6')
                fc6 = tf.add(tf.matmul(flatten_input, w_fc6), b_fc6)
                relu_fc6 = tf.nn.relu(fc6)
                dropout_fc6 = tf.nn.dropout(relu_fc6, self.keep_prob)

            with tf.name_scope('fc7') as scope:
                w_fc7 = weight_init(shape=[4096,  4096], stddev=0.01, name='w_fc7')
                b_fc7 = bias_init(0.0, [4096], 'b_fc7')
                fc7 = tf.add(tf.matmul(dropout_fc6, w_fc7), b_fc7)
                relu_fc7 = tf.nn.relu(fc7)
                dropout_fc7 = tf.nn.dropout(relu_fc7, self.keep_prob)

            with tf.name_scope('softmax') as scope:
                w_softmax = weight_init(shape=[4096,  1000], stddev=0.01, name='w_softmax')
                b_softmax = bias_init(0.0, [1000], 'b_softmax')
                scores = tf.add(tf.matmul(dropout_fc7, w_softmax), b_softmax)
                output = tf.nn.softmax(scores)

        return output


