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
    """Defines a fully-connected layer
    :param _input: The input batch to be processed
    :param size: Defines the number of neurons into the layer
    :param name: Defines de name of the layer
    :param activation: Name of the activation function to be used
    :param activation_param: Defines the parameter to be used for some activation functions

    :returns: The activation values of the neurons
    """

    # shape is a 1D tensor with 4 values
    num_features_in = _input.get_shape().as_list()[-1]

    # reshape to  1D vector
    shape = [num_features_in, size]
    W = tf.Variable(gaussian_weights(shape, 0.0, 0.02), name=name)
    b = tf.Variable(tf.zeros(size))

    # computes input[N_in x D] x W[N_in x N_out] + b[N_out]
    layer = tf.add(tf.matmul(_input, W), b)

    if activation == 'sigmoid':
        return tf.nn.sigmoid(layer)
    elif activation == 'tanh':
        return tf.nn.tanh(layer)
    elif activation == 'relu':
        return tf.nn.relu(layer)
    elif activation == 'leaky-relu':
        return tf.nn.leaky_relu(layer, alpha=activation_param)
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


def eval_confusion_matrix(labels, predictions, num_classes):
    con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_classes)

    con_matrix_sum = tf.Variable(tf.zeros(shape=(num_classes, num_classes), dtype=tf.int32),
                                 trainable=False,
                                 name="confusion_matrix_result",
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES])

    update_op = tf.assign_add(con_matrix_sum, con_matrix)

    return tf.convert_to_tensor(con_matrix_sum), update_op


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

    net = mlp_fn(features,
                 params['input_size'],
                 params['number_of_classes'],
                 params['activation'],
                 params['activation_param'])
    logits = net["output"]

    idx_predicted_class = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={'class_ids': idx_predicted_class[:, tf.newaxis],
                         'probabilities': tf.nn.softmax(logits),
                         'logits': logits})
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
            cm_op = eval_confusion_matrix(labels=idx_true_class, predictions=idx_predicted_class,
                                          num_classes=params['number_of_classes'])

            # ''' confusion matrix summaries '''
            # sess = tf.Session()
            # correct_labels = [params['class_labels'][p] for p in tf.map_fn(lambda x: x, idx_true_class)]
            # predict_labels = [params['class_labels'][p] for p in tf.map_fn(lambda x: x, idx_predicted_class)]
            # abs_img_summary = cm.plot_confusion_matrix(correct_labels=correct_labels,
            #                                            predict_labels=predict_labels,
            #                                            labels=params['class_labels'],
            #                                            tensor_name='abs-confusion-matrix')
            # norm_img_summary = cm.plot_confusion_matrix(correct_labels=correct_labels,
            #                                             predict_labels=predict_labels,
            #                                             labels=params['class_labels'],
            #                                             tensor_name='norm-confusion-matrix',
            #                                             normalize=True)
            # tf.summary.image(abs_img_summary)
            # tf.summary.image(norm_img_summary)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=idx_predicted_class,
                loss=loss,
                eval_metric_ops={'accuracy': acc_op, 'precision': precision_op, 'recall': recall_op,
                                 'false_neg': false_neg_op, 'conf_matrix': cm_op})
        else:  # TRAIN
            if params['optimizer'] == 'gd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
            elif params['optimizer'] == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'],
                                                       momentum=params['opt_param'])
            elif params['optimizer'] == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            elif params['optimizer'] == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)
