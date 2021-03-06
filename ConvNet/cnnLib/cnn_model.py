#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:29:43 2018

@author: Jose M. Saavedra
This module contains the model specifications
"""
import os
import tensorflow as tf
# from official.resnet import cifar10_main
from . import cnn_arch as arch


def initializedModel(model_dir):
    return os.path.exists(model_dir + '/init.init')


# create a file indicating that init was done
def saveInitializationIndicator(model_dir):
    with open(model_dir + '/init.init', 'w+') as f:
        f.write('1')

    # defining a model that feeds the Estimator


def eval_confusion_matrix(labels, predictions, num_classes):
    con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes)

    con_matrix_sum = tf.Variable(tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                                 trainable=False,
                                 name="confusion_matrix_result",
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES])

    update_op = tf.assign_add(con_matrix_sum, con_matrix)

    return tf.convert_to_tensor(con_matrix_sum), update_op


def initialize_net(features, params, is_training):
    if params['arch'] == 'MNIST':
        return arch.mnistnet_fn(features, params['image_shape'], params['number_of_classes'],
                               params['number_of_channels'], is_training)
    elif params['arch'] == 'VGG16':
        return arch.vgg16_net_fn(features, params['image_shape'], params['number_of_classes'],
                                params['number_of_channels'], is_training)
    elif params['arch'] == 'SIMPLE_VGG':
        return arch.simple_vgg_net_fn(features, params['image_shape'], params['number_of_classes'],
                                     params['number_of_channels'], is_training)
    elif params['arch'] == 'SIMPLE_VGG2':
        return arch.simple_vgg2_net_fn(features, params['image_shape'], params['number_of_classes'],
                                      params['number_of_channels'], is_training)
    else:
        raise ValueError("network architecture is unknown")


def model_fn(features, labels, mode, params):
    """Defines a model that feeds the Estimator
    The signature here is standard according to Estimators.
    The output is an EstimatorSpec
    :param features: The set of features to be processed
    :param labels: The set of correct labels for the features
    :param mode: Specifies if this training, evaluation or prediction
    :param params: dict of hyperparameters to configure Estimator from hyper parameter tuning

    :returns: The model definition with the ops and objects to be run by an Estimator
    :rtype: EstimatorSpec
    """

    if mode == tf.estimator.ModeKeys.TRAIN:
        is_training = True
    else:
        is_training = False

    net = initialize_net(features, params, is_training)

    logits = net["output"]

    idx_predicted_class = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'class_ids': idx_predicted_class[:, tf.newaxis],
                         'probabilities': tf.nn.softmax(logits),
                         'logits': logits, 'deep_features': net[params['feats_layer']]})
    else:
        # Define loss - e.g. cross_entropy - mean(cross_entropy x batch)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        if mode == tf.estimator.ModeKeys.EVAL:
            idx_true_class = tf.argmax(labels, 1)

            # Define the evaluation metrics of the model
            acc_op = tf.metrics.accuracy(labels=idx_true_class, predictions=idx_predicted_class)
            precision_op = tf.metrics.precision(labels=idx_true_class, predictions=idx_predicted_class)
            recall_op = tf.metrics.recall(labels=idx_true_class, predictions=idx_predicted_class)
            false_neg_op = tf.metrics.false_negatives(labels=idx_true_class, predictions=idx_predicted_class)
            # cm_op = eval_confusion_matrix(labels=idx_true_class, predictions=idx_predicted_class,
            #                               num_classes=params['number_of_classes'])

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=idx_predicted_class,
                loss=loss,
                eval_metric_ops={'accuracy': acc_op, 'precision': precision_op, 'recall': recall_op,
                                 'false_neg': false_neg_op,  # 'conf_matrix': cm_op
                                 })
        else:  # TRAIN
            # model initialization if params[ckpt] is defined. This is used for fine-tuning
            if not initializedModel(params['model_dir']):
                variables = tf.trainable_variables()
                print(variables)
                if 'ckpt' in params and params['ckpt'] is not None:
                    print('---Loading checkpoint : ' + params['ckpt'])
                    """
                    assignment_map is very critical for fine-tunning
                    this must be a dictionary mapping
                    checkpoint_scope_name/variable_name : scope_name/variable
                    """
                    tf.train.init_from_checkpoint(ckpt_dir_or_file=params['ckpt'],
                                                  assignment_map={v.name.split(':')[0]: v for v in variables})

                    # save and indicator file
                    saveInitializationIndicator(params['model_dir'])
                    print('---Checkpoint : ' + params['ckpt'] + ' was loaded')

            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS)  # to allow update [is_training variable] used by batch_normalization
            with tf.control_dependencies(update_ops):
                if params['lr_decay'] == 'exponential':
                    lr = tf.train.exponential_decay(learning_rate=params['learning_rate'],
                                                    global_step=tf.train.get_global_step(),
                                                    decay_steps=params['lr_decay_steps'],
                                                    decay_rate=0.96, staircase=True)
                elif params['lr_decay'] == 'polynomial':
                    end_lr = params['learning_rate'] / 10
                    lr = tf.train.polynomial_decay(learning_rate=params['learning_rate'],
                                                   global_step=tf.train.get_global_step(),
                                                   decay_steps=params['lr_decay_steps'], end_learning_rate=end_lr)
                else:
                    lr = params['learning_rate']

                tf.summary.scalar('learning_rate', lr)

                if params['optimizer'] == 'gd':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
                elif params['optimizer'] == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=params['opt_param'])
                elif params['optimizer'] == 'adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
                elif params['optimizer'] == 'rmsprop':
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
                else:
                    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)
