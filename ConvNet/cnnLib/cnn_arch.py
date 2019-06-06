#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:38:03 2018

@author: Jose M. Saavedra
In this file, different architectures are defined
"""

import tensorflow as tf
from . import layers


# %%
# A net for sketch classification, this is similar to AlexNet
# features: containing feature vectors to be trained
# input_shape: [height, width]
# n_classes int
# is_training: True for training and False for testing
def mnistnet_fn(features, input_shape, n_classes, n_channels, is_training=True):
    with tf.variable_scope("net_scope"):
        # reshape input to fit a  4D tensor
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1], n_channels])

        conv_1 = layers.conv_layer(x_tensor, shape=[3, 3, n_channels, 32], stride=1, name='conv_1',
                                   is_training=is_training)  # 256
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))

        pool_1 = layers.max_pool_layer(conv_1, 3, 2)  # 14x14
        print(" pool_1: {} ".format(pool_1.get_shape().as_list()))

        conv_2 = layers.conv_layer(pool_1, shape=[3, 3, 32, 64], name='conv_2', is_training=is_training)
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))

        pool_2 = layers.max_pool_layer(conv_2, 3, 2)  # 7x7
        print(" pool_2: {} ".format(pool_2.get_shape().as_list()))

        fc1 = layers.fc_layer(pool_2, 256, name='fc3')
        print(" fc1: {} ".format(fc1.get_shape().as_list()))

    with tf.variable_scope("class_layer"):
        fc2 = layers.fc_layer(fc1, n_classes, name='fc4', use_relu=False)
        print(" fc2: {} ".format(fc2.get_shape().as_list()))

    return {'output': fc2, 'fc1': fc1}


def simple_vgg_net_fn(features, input_shape, n_classes, n_channels, is_training=True):
    with tf.variable_scope("net_scope"):
        # reshape input to fit a  4D tensor
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1], n_channels])

        # 64-channels block
        conv_1 = layers.conv_layer(x_tensor, shape=[3, 3, n_channels, 64], stride=1, name='conv1-3-64',
                                   is_training=is_training)
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))

        conv_2 = layers.conv_layer(conv_1, shape=[3, 3, 64, 64], stride=1, name='conv2-3-64',
                                   is_training=is_training)
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))

        pool_1 = layers.max_pool_layer(conv_2, 3, 2)
        print(" pool_1: {} ".format(pool_1.get_shape().as_list()))

        # 128-channels block
        conv_3 = layers.conv_layer(pool_1, shape=[3, 3, 64, 128], stride=1, name='conv3-3-128',
                                   is_training=is_training)
        print(" conv_3: {} ".format(conv_3.get_shape().as_list()))

        conv_4 = layers.conv_layer(conv_3, shape=[3, 3, 128, 128], stride=1, name='conv4-3-128',
                                   is_training=is_training)
        print(" conv_4: {} ".format(conv_4.get_shape().as_list()))

        pool_2 = layers.max_pool_layer(conv_4, 3, 2)
        print(" pool_2: {} ".format(pool_2.get_shape().as_list()))

        # 256-channels block
        conv_5 = layers.conv_layer(pool_2, shape=[3, 3, 128, 256], stride=1, name='conv5-3-256',
                                   is_training=is_training)
        print(" conv_5: {} ".format(conv_5.get_shape().as_list()))

        conv_6 = layers.conv_layer(conv_5, shape=[3, 3, 256, 256], stride=1, name='conv6-3-256',
                                   is_training=is_training)
        print(" conv_6: {} ".format(conv_6.get_shape().as_list()))

        pool_3 = layers.max_pool_layer(conv_6, 3, 2)
        print(" pool_3: {} ".format(pool_3.get_shape().as_list()))

        # fully-connected layers block
        fc1 = layers.fc_layer(pool_3, 1024, name='fc1-1024')
        print(" fc1: {} ".format(fc1.get_shape().as_list()))
        fc2 = layers.fc_layer(fc1, 1024, name='fc2-1024')
        print(" fc2: {} ".format(fc2.get_shape().as_list()))

    with tf.variable_scope("class_layer"):
        fc3 = layers.fc_layer(fc2, n_classes, name='fc3-{}'.format(n_channels), use_relu=False)
        print(" fc3: {} ".format(fc3.get_shape().as_list()))

    return {'output': fc3, 'fc1': fc1, 'fc2': fc2}


def simple_vgg2_net_fn(features, input_shape, n_classes, n_channels, is_training=True):
    with tf.variable_scope("net_scope"):
        # reshape input to fit a  4D tensor
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1], n_channels])

        # 64-channels block
        conv_1 = layers.conv_layer(x_tensor, shape=[3, 3, n_channels, 64], stride=1, name='conv1-3-64',
                                   is_training=is_training)
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))

        # conv_2 = layers.conv_layer(conv_1, shape=[3, 3, 64, 64], stride=1, name='conv2-3-64',
        #                            is_training=is_training)
        # print(" conv_2: {} ".format(conv_2.get_shape().as_list()))

        pool_1 = layers.max_pool_layer(conv_1, 3, 2)
        print(" pool_1: {} ".format(pool_1.get_shape().as_list()))

        # 128-channels block
        conv_3 = layers.conv_layer(pool_1, shape=[3, 3, 64, 128], stride=1, name='conv3-3-128',
                                   is_training=is_training)
        print(" conv_3: {} ".format(conv_3.get_shape().as_list()))

        # conv_4 = layers.conv_layer(conv_3, shape=[3, 3, 128, 128], stride=1, name='conv4-3-128',
        #                            is_training=is_training)
        # print(" conv_4: {} ".format(conv_4.get_shape().as_list()))

        pool_2 = layers.max_pool_layer(conv_3, 3, 2)
        print(" pool_2: {} ".format(pool_2.get_shape().as_list()))

        # 256-channels block
        conv_5 = layers.conv_layer(pool_2, shape=[3, 3, 128, 256], stride=1, name='conv5-3-256',
                                   is_training=is_training)
        print(" conv_5: {} ".format(conv_5.get_shape().as_list()))

        # conv_6 = layers.conv_layer(conv_5, shape=[3, 3, 256, 256], stride=1, name='conv6-3-256',
        #                            is_training=is_training)
        # print(" conv_6: {} ".format(conv_6.get_shape().as_list()))

        pool_3 = layers.max_pool_layer(conv_5, 3, 2)
        print(" pool_3: {} ".format(pool_3.get_shape().as_list()))

        # fully-connected layers block
        fc1 = layers.fc_layer(pool_3, 1024, name='fc1-1024')
        print(" fc1: {} ".format(fc1.get_shape().as_list()))
        fc2 = layers.fc_layer(fc1, 1024, name='fc2-1024')
        print(" fc2: {} ".format(fc2.get_shape().as_list()))

    with tf.variable_scope("class_layer"):
        fc3 = layers.fc_layer(fc2, n_classes, name='fc3-{}'.format(n_channels), use_relu=False)
        print(" fc3: {} ".format(fc3.get_shape().as_list()))

    return {'output': fc3, 'fc1': fc1, 'fc2': fc2}


def vgg16_net_fn(features, input_shape, n_classes, n_channels, is_training=True):
    with tf.variable_scope("net_scope"):
        # reshape input to fit a  4D tensor
        x_tensor = tf.reshape(features, [-1, input_shape[0], input_shape[1], n_channels])

        # 64-channels block
        conv_1 = layers.conv_layer(x_tensor, shape=[3, 3, n_channels, 64], stride=1, name='conv1-3-64',
                                   is_training=is_training)
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))

        conv_2 = layers.conv_layer(conv_1, shape=[3, 3, 64, 64], stride=1, name='conv2-3-64',
                                   is_training=is_training)
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))

        pool_1 = layers.max_pool_layer(conv_2, 3, 2)
        print(" pool_1: {} ".format(pool_1.get_shape().as_list()))

        # 128-channels block
        conv_3 = layers.conv_layer(pool_1, shape=[3, 3, 64, 128], stride=1, name='conv3-3-128',
                                   is_training=is_training)
        print(" conv_3: {} ".format(conv_3.get_shape().as_list()))

        conv_4 = layers.conv_layer(conv_3, shape=[3, 3, 128, 128], stride=1, name='conv4-3-128',
                                   is_training=is_training)
        print(" conv_4: {} ".format(conv_4.get_shape().as_list()))

        pool_2 = layers.max_pool_layer(conv_4, 3, 2)
        print(" pool_2: {} ".format(pool_2.get_shape().as_list()))

        # 256-channels block
        conv_5 = layers.conv_layer(pool_2, shape=[3, 3, 128, 256], stride=1, name='conv5-3-256',
                                   is_training=is_training)
        print(" conv_5: {} ".format(conv_5.get_shape().as_list()))

        conv_6 = layers.conv_layer(conv_5, shape=[3, 3, 256, 256], stride=1, name='conv6-3-256',
                                   is_training=is_training)
        print(" conv_6: {} ".format(conv_6.get_shape().as_list()))

        conv_7 = layers.conv_layer(conv_6, shape=[3, 3, 256, 256], stride=1, name='conv7-3-256',
                                   is_training=is_training)
        print(" conv_7: {} ".format(conv_7.get_shape().as_list()))

        pool_3 = layers.max_pool_layer(conv_7, 3, 2)
        print(" pool_3: {} ".format(pool_3.get_shape().as_list()))

        # 512-channels block1
        conv_8 = layers.conv_layer(pool_3, shape=[3, 3, 256, 512], stride=1, name='conv8-3-512',
                                   is_training=is_training)
        print(" conv_8: {} ".format(conv_8.get_shape().as_list()))

        conv_9 = layers.conv_layer(conv_8, shape=[3, 3, 512, 512], stride=1, name='conv9-3-512',
                                   is_training=is_training)
        print(" conv_9: {} ".format(conv_9.get_shape().as_list()))

        conv_10 = layers.conv_layer(conv_9, shape=[3, 3, 512, 512], stride=1, name='conv10-3-512',
                                    is_training=is_training)
        print(" conv_10: {} ".format(conv_10.get_shape().as_list()))

        pool_4 = layers.max_pool_layer(conv_10, 3, 2)
        print(" pool_4: {} ".format(pool_4.get_shape().as_list()))

        # 512-channels block2
        conv_11 = layers.conv_layer(pool_4, shape=[3, 3, 512, 512], stride=1, name='conv11-3-512',
                                    is_training=is_training)
        print(" conv_11: {} ".format(conv_11.get_shape().as_list()))

        conv_12 = layers.conv_layer(conv_11, shape=[3, 3, 512, 512], stride=1, name='conv12-3-512',
                                    is_training=is_training)
        print(" conv_12: {} ".format(conv_12.get_shape().as_list()))

        conv_13 = layers.conv_layer(conv_12, shape=[3, 3, 512, 512], stride=1, name='conv13-3-512',
                                    is_training=is_training)
        print(" conv_13: {} ".format(conv_13.get_shape().as_list()))

        pool_5 = layers.max_pool_layer(conv_13, 3, 2)
        print(" pool_5: {} ".format(pool_5.get_shape().as_list()))

        # fully-connected layers block
        fc1 = layers.fc_layer(pool_5, 4096, name='fc1-4096')
        print(" fc1: {} ".format(fc1.get_shape().as_list()))
        fc2 = layers.fc_layer(fc1, 4096, name='fc2-4096')
        print(" fc2: {} ".format(fc2.get_shape().as_list()))

    with tf.variable_scope("class_layer"):
        fc3 = layers.fc_layer(fc2, n_classes, name='fc3-{}'.format(n_channels), use_relu=False)
        print(" fc3: {} ".format(fc3.get_shape().as_list()))

    return {'output': fc3, 'fc1': fc1, 'fc2': fc2}
