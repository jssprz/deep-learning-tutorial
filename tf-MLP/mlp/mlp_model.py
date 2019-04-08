'''
Created on Mar 4, 2019

@author: jsaavedr
'''

import tensorflow as tf


def gaussian_weights(shape, mean, stddev):
    return tf.truncated_normal(shape,
                               mean=mean,
                               stddev=stddev)


def fc_layer(_input, size, name, activation='sigmoid', activation_param=0.1):
    """
    a fully connected layer
    """

    # shape is a 1D tensor with 4 values
    num_features_in = _input.get_shape().as_list()[-1]

    # reshape to  1D vector
    shape = [num_features_in, size]
    W = tf.Variable(gaussian_weights(shape, 0.0, 0.02), name=name)
    b = tf.Variable(tf.zeros(size))

    # just a  multiplication between input[N_in x D]xW[N_in x N_out]
    layer = tf.add(tf.matmul(_input, W), b)

    if activation == 'sigmoid':
        layer = tf.nn.sigmoid(layer)
    if activation == 'tanh':
        layer = tf.nn.tanh(layer)
    if activation == 'relu':
        layer = tf.nn.relu(layer)
    if activation == 'leaky-relu':
        layer = tf.nn.leaky_relu(layer, alpha=activation_param)

    return layer


def mlp_fn(features, input_size, n_classes, activation, activation_param):
    with tf.variable_scope("mlp_scope"):
        # input
        features = tf.reshape(features, [-1, input_size])
        print(" features {} ".format(features.get_shape().as_list()))

        # fully connected layer 1
        fc1 = fc_layer(features, 100, name='fc1', activation=activation, activation_param=activation_param)
        print(" fc1: {} ".format(fc1.get_shape().as_list()))

        # fully connected layer 2
        fc2 = fc_layer(fc1, 100, name='fc2', activation=activation, activation_param=activation_param)
        print(" fc2: {} ".format(fc2.get_shape().as_list()))

        # fully connected layer 3, output layer
        fc3 = fc_layer(fc2, n_classes, name='fc3', activation='')
        print(" fc3: {} ".format(fc3.get_shape().as_list()))

    return {"output": fc3}


# defining a model that feeds the Estimator
def model_fn(features, labels, mode, params):
    """The signature here is standard according to Estimators. 
       The output is an EstimatorSpec
    """

    net = mlp_fn(features,
                 params['input_size'],
                 params['number_of_classes'],
                 params['activation'],
                 params['activation-param'])
    logits = net["output"]

    idx_predicted_class = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # --------------------------------------
        # If prediction mode, predictions is returned
        predictions = {
            'class_ids': idx_predicted_class[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:  # TRAIN or EVAL
        idx_true_class = tf.argmax(labels, 1)

        # Define the evaluation metrics of the model
        acc_op, accuracy_update = tf.metrics.accuracy(labels=idx_true_class, predictions=idx_predicted_class)
        recall_op = tf.metrics.recall(labels=idx_true_class, predictions=idx_predicted_class)
        false_neg_op = tf.metrics.false_negatives(labels=idx_true_class, predictions=idx_predicted_class)

        # Define loss - e.g. cross_entropy - mean(cross_entropy x batch)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)

        if params['optimizer'] == 'gradient-descent':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        # EstimatorSpec
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=idx_predicted_class,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={'accuracy': (acc_op, accuracy_update), 'recall': recall_op, 'false_neg': false_neg_op})
