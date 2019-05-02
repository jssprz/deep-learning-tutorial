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

        conv_1 = layers.conv_layer(x_tensor, shape=[3, 3, n_channels, 16], stride=1, name='conv_1',
                                   is_training=is_training)  # 256
        print(" conv_1: {} ".format(conv_1.get_shape().as_list()))

        pool_1 = layers.max_pool_layer(conv_1, 3, 2)  # 14x14
        print(" pool_1: {} ".format(pool_1.get_shape().as_list()))

        conv_2 = layers.conv_layer(conv_1, shape=[3, 3, 16, 32], name='conv_2', is_training=is_training)
        print(" conv_2: {} ".format(conv_2.get_shape().as_list()))

        pool_2 = layers.max_pool_layer(conv_2, 3, 2)  # 7x7
        print(" pool_2: {} ".format(pool_2.get_shape().as_list()))

        fc3 = layers.fc_layer(pool_2, 256, name='fc3')
        print(" fc3: {} ".format(fc3.get_shape().as_list()))

    with tf.variable_scope("class_layer"):
        fc4 = layers.fc_layer(fc3, n_classes, name='fc4', use_relu=False)
        print(" fc4: {} ".format(fc4.get_shape().as_list()))

    return {"output": fc4}
