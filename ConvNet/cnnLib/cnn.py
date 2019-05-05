#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 14:39:56 2018
@author: jose.saavedra

This implements basic operations on a cnn

"""

import os
import numpy as np
import tensorflow as tf
from . import configuration as conf
from . import cnn_model as model
from . import data as data
from . import imgproc
from . import confusion_matrix as cm


class CNN:
    def __init__(self, str_config, params):
        self.initFromCkpt = False
        self.ckpt_file = None
        if 'ckpt' in params:
            self.initFromCkpt = True
            self.ckpt_file = params['ckpt']
        # reading configuration file
        self.configuration = conf.ConfigurationFile(str_config, params['modelname'])
        self.modelname = self.configuration.model_name
        self.device = params['device']
        self.processFun = imgproc.getProcessFun(self.configuration.process_fun)
        # validatitn snapShotDir
        assert os.path.exists(self.configuration.snapshot_dir), "Path {} does not exist".format(
            self.configuration.snapshot_dir)
        # if not os.path.exists(os.path.dirname(self.configuration.snapshot_dir)) :
        #    os.makedirs(os.path.dirname(self.configuration.snapshot_dir))
        # metadata
        filename_mean = os.path.join(self.configuration.data_dir, "mean.dat")
        metadata_file = os.path.join(self.configuration.data_dir, "metadata.dat")
        # reading metadata
        self.image_shape = np.fromfile(metadata_file, dtype=np.int32)
        #
        print("image shape: {}".format(self.image_shape))
        # load mean
        mean_img = np.fromfile(filename_mean, dtype=np.float32)
        self.mean_img = np.reshape(mean_img, self.image_shape.tolist())
        # defining files for training and test
        self.filename_train = os.path.join(self.configuration.data_dir, "train.tfrecords")
        self.filename_test = os.path.join(self.configuration.data_dir, "test.tfrecords")
        # print(" mean {}".format(self.mean_img.shape))

    def train(self):
        """training"""
        # -using device gpu or cpu
        with tf.device(self.device):
            estimator_config = tf.estimator.RunConfig(model_dir=self.configuration.snapshot_dir,
                                                      save_checkpoints_steps=self.configuration.save_checkpoints_steps,
                                                      keep_checkpoint_max=0)
            classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                                config=estimator_config,
                                                params={'learning_rate': self.configuration.learning_rate,
                                                        'number_of_classes': self.configuration.number_of_classes,
                                                        'image_shape': self.image_shape,
                                                        'number_of_channels': self.configuration.number_of_channels,
                                                        'model_dir': self.configuration.snapshot_dir,
                                                        'ckpt': self.ckpt_file,
                                                        'arch': self.configuration.arch,
                                                        'optimizer': self.configuration.optimizer,
                                                        'batch_size': self.configuration.batch_size,
                                                        'lr_decay': self.configuration.lr_decay,
                                                        'lr_decay_steps': self.configuration.lr_decay_steps
                                                        }
                                                )
            #
            tf.logging.set_verbosity(tf.logging.INFO)  # Just to have some logs to display for demonstration
            # training
            train_spec = tf.estimator.TrainSpec(input_fn=lambda: data.input_fn(self.filename_train,
                                                                               self.image_shape,
                                                                               self.mean_img,
                                                                               is_training=True,
                                                                               configuration=self.configuration),
                                                max_steps=self.configuration.number_of_iterations)
            # max_steps is not useful when inherited checkpoint is used
            eval_spec = tf.estimator.EvalSpec(input_fn=lambda: data.input_fn(self.filename_test,
                                                                             self.image_shape,
                                                                             self.mean_img,
                                                                             is_training=False,
                                                                             configuration=self.configuration),
                                              throttle_secs=1)

            tf.summary.text('model-hyper-parameters', tf.convert_to_tensor(self.configuration.__str__()))

            tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    def test(self):
        """test checkpoint exist """
        assert os.path.exists(os.path.join(self.configuration.snapshot_dir,
                                           "checkpoint")), "Checkpoint file does not exist in {}".format(
            self.configuration.snapshot_dir)
        """testing"""
        with tf.device(self.device):
            estimator_config = tf.estimator.RunConfig(model_dir=self.configuration.snapshot_dir,
                                                      save_checkpoints_steps=self.configuration.save_checkpoints_steps,
                                                      keep_checkpoint_max=10)
            classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                                config=estimator_config,
                                                params={'learning_rate': self.configuration.learning_rate,
                                                        'number_of_classes': self.configuration.number_of_classes,
                                                        'image_shape': self.image_shape,
                                                        'number_of_channels': self.configuration.number_of_channels,
                                                        'model_dir': self.configuration.snapshot_dir,
                                                        'ckpt': self.ckpt_file,
                                                        'arch': self.configuration.arch
                                                        }
                                                )
            result = classifier.evaluate(input_fn=lambda: data.input_fn(self.filename_test,
                                                                        self.image_shape,
                                                                        self.mean_img,
                                                                        is_training=False,
                                                                        configuration=self.configuration),
                                         checkpoint_path=self.ckpt_file)
            print(result)

    def confusion_matrix_for_test(self, checkpoint_iter=None):
        """

        :param checkpoint_iter:
        :return:
        """
        # test checkpoint exist
        assert os.path.exists(os.path.join(self.configuration.snapshot_dir, "checkpoint")), \
            "Checkpoint file does not exist in {}".format(self.configuration.snapshot_dir)

        with tf.device(self.device):
            tf.logging.set_verbosity(tf.logging.INFO)

            classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                                model_dir=self.configuration.snapshot_dir,
                                                params={'learning_rate': self.configuration.learning_rate,
                                                        'number_of_classes': self.configuration.number_of_classes,
                                                        'image_shape': self.image_shape,
                                                        'number_of_channels': self.configuration.number_of_channels,
                                                        'arch': self.configuration.arch
                                                        })

            with open(os.path.join(self.configuration.data_dir, "test.txt"), 'r') as file:
                lines = [line.rstrip() for line in file]
                lines_ = [tuple(line.rstrip().split('\t')) for line in lines]
                filenames, labels = zip(*lines_)
                truth_labels = [self.configuration.class_labels[x] for x in data.validateLabels(labels)]

            summary_dir = os.path.join(self.configuration.snapshot_dir, 'cm')
            if 'cm' not in os.listdir(self.configuration.snapshot_dir):
                os.mkdir(os.path.join(self.configuration.snapshot_dir, 'cm'))

            summary_writer = tf.summary.FileWriter(logdir=summary_dir, max_queue=150)

            if checkpoint_iter is not None:
                result = list(classifier.predict(
                    input_fn=lambda: data.input_fn(self.filename_test, self.image_shape, self.mean_img, False,
                                                   self.configuration),
                    checkpoint_path=os.path.join(self.configuration.snapshot_dir,
                                                 'model.ckpt-{}'.format(checkpoint_iter))))

                predicted_labels = [self.configuration.class_labels[p["class_ids"][0]] for p in result]

                ''' confusion matrix summaries '''
                abs_img_summary = cm.plot_confusion_matrix(correct_labels=truth_labels,
                                                           predict_labels=predicted_labels,
                                                           labels=self.configuration.class_labels,
                                                           tensor_name='abs-confusion-matrix')
                norm_img_summary = cm.plot_confusion_matrix(correct_labels=truth_labels,
                                                            predict_labels=predicted_labels,
                                                            labels=self.configuration.class_labels,
                                                            tensor_name='norm-confusion-matrix',
                                                            normalize=True)
                summary_writer.add_summary(abs_img_summary)
                summary_writer.add_summary(norm_img_summary)
            else:
                checkpoints_iters = sorted([int(x[11:-6]) for x in filter(lambda s: '.index' in s,
                                                                          os.listdir(self.configuration.snapshot_dir))])
                for checkpoint_iter in checkpoints_iters:
                    result = list(classifier.predict(
                        input_fn=lambda: data.input_fn(self.filename_test, self.image_shape, self.mean_img, False,
                                                       self.configuration),
                        checkpoint_path=os.path.join(self.configuration.snapshot_dir,
                                                     'model.ckpt-{}'.format(checkpoint_iter))))

                    predicted_labels = [self.configuration.class_labels[p["class_ids"][0]] for p in result]

                    ''' confusion matrix summaries '''
                    abs_img_summary = cm.plot_confusion_matrix(correct_labels=truth_labels,
                                                               predict_labels=predicted_labels,
                                                               labels=self.configuration.class_labels,
                                                               tensor_name='abs-confusion-matrix')
                    norm_img_summary = cm.plot_confusion_matrix(correct_labels=truth_labels,
                                                                predict_labels=predicted_labels,
                                                                labels=self.configuration.class_labels,
                                                                tensor_name='norm-confusion-matrix',
                                                                normalize=True)
                    summary_writer.add_summary(abs_img_summary, checkpoint_iter)
                    summary_writer.add_summary(norm_img_summary, checkpoint_iter)

    def predict(self, filename):
        """test checkpoint exist """
        assert os.path.exists(os.path.join(self.configuration.snapshot_dir,
                                           "checkpoint")), "Checkpoint file does not exist in {}".format(
            self.configuration.snapshot_dir)
        classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                            model_dir=self.configuration.snapshot_dir,
                                            params={'learning_rate': self.configuration.learning_rate,
                                                    'number_of_classes': self.configuration.number_of_classes,
                                                    'image_shape': self.image_shape,
                                                    'number_of_channels': self.configuration.number_of_channels,
                                                    'arch': self.configuration.arch
                                                    })
        #
        tf.logging.set_verbosity(tf.logging.INFO)  # Just to have some logs to display for demonstration

        input_image = data.input_fn_for_prediction(filename,
                                                   self.image_shape,
                                                   self.mean_img,
                                                   self.configuration.number_of_channels,
                                                   self.processFun)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=input_image,
            num_epochs=1,
            shuffle=False
        )
        # classifier could use checkpoint_path to define the checkpoint to be used
        predicted_result = list(classifier.predict(input_fn=predict_input_fn))
        return predicted_result

    def predict_on_list(self, list_of_images):
        """test checkpoint exist """
        assert os.path.exists(os.path.join(self.configuration.snapshot_dir,
                                           "checkpoint")), "Checkpoint file does not exist in {}".format(
            self.configuration.snapshot_dir)
        classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                            model_dir=self.configuration.snapshot_dir,
                                            params={'learning_rate': self.configuration.learning_rate,
                                                    'number_of_classes': self.configuration.number_of_classes,
                                                    'image_shape': self.image_shape,
                                                    'number_of_channels': self.configuration.number_of_channels,
                                                    'arch': self.configuration.arch
                                                    })
        #
        tf.logging.set_verbosity(tf.logging.INFO)  # Just to have some logs to display for demonstration
        batch_of_images = data.input_fn_for_prediction_on_list(list_of_images,
                                                               self.image_shape,
                                                               self.mean_img,
                                                               self.configuration.number_of_channels,
                                                               self.processFun)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=batch_of_images,
            num_epochs=1,
            shuffle=False)
        # classifier could use checkpoint_path to define the checkpoint to be used
        predicted_result = list(classifier.predict(input_fn=predict_input_fn, yield_single_examples=False))
        return predicted_result

    def save_model(self):
        """save model for prediction """
        assert os.path.exists(os.path.join(self.configuration.snapshot_dir,
                                           "checkpoint")), "Checkpoint file does not exist in {}".format(
            self.configuration.snapshot_dir)
        classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                            model_dir=self.configuration.snapshot_dir,
                                            params={'learning_rate': self.configuration.learning_rate,
                                                    'number_of_classes': self.configuration.number_of_classes,
                                                    'image_shape': self.image_shape,
                                                    'number_of_channels': self.configuration.number_of_channels,
                                                    'arch': self.configuration.arch
                                                    })

        #
        def serving_input_receiver_fn():
            feat_spec = tf.placeholder(dtype=tf.float32,
                                       shape=[None, self.image_shape[0], self.image_shape[1], self.image_shape[2]])
            return tf.estimator.export.TensorServingInputReceiver(feat_spec, feat_spec)

        str_model = classifier.export_saved_model(self.configuration.snapshot_dir, serving_input_receiver_fn)
        final_str_model = os.path.join(self.configuration.data_dir, "cnn-model")
        os.rename(str_model, final_str_model)
        print("The models was successfully saved at {}".format(final_str_model))
