#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

from configparser import ConfigParser


class ConfigurationFile:
    """
     An instance of ConfigurationFile contains required parameters to train a 
     convolutional neural network
    """

    def __init__(self, config_path, section_name):
        self.__config = ConfigParser()
        self.__config.read(config_path)

        try:
            section = self.__config[section_name]
        except Exception:
            raise ValueError(" {} is not a valid section".format(section_name))

        try:
            self.__modelname = section_name
            self.__arch = section["ARCH"]
            self.__snapshot_prefix = section['SNAPSHOT_DIR']
            self.__process_fun = 'default'
            self.__data_dir = section['DATA_DIR']
            self.__optimizer = section['OPTIMIZER']
            if 'PROCESS_FUN' in section:
                self.__process_fun = section['PROCESS_FUN']

            self.__number_of_classes = int(section['NUM_CLASSES'])
            self.__number_of_iterations = int(section['NUM_ITERATIONS'])
            self.__dataset_size = int(section['DATASET_SIZE'])
            self.__test_size = int(section['TEST_SIZE'])
            self.__batch_size = int(section['BATCH_SIZE'])
            self.__save_checkpoints_steps = int(section['SAVE_CHECKPOINTS_STEPS'])
            self.__test_time = int(section['TEST_TIME'])

            self.__lr = float(section['LEARNING_RATE'])
            if 'LR_DECAY' in section:
                self.__lr_decay = section['LR_DECAY']
                self.__lr_decay_steps = float(section['DECAY_STEPS']) if 'DECAY_STEPS' in section else 10000

            self.__estimated_number_of_batches = int(float(self.__dataset_size) / float(self.__batch_size))
            self.__estimated_number_of_batches_test = int(float(self.__test_size) / float(self.__batch_size))
            self.__number_of_epochs = int(float(self.__number_of_iterations) / float(self.__estimated_number_of_batches))
            self.__channels = int(section['CHANNELS'])
            self.__class_labels = section['CLASS_LABELS'].split(',')

            if 'DEEP_FEATS_LAYER' in section:
                self.__deep_feats_layer = section['DEEP_FEATS_LAYER']

            if 'METRIC' in section:
                self.__metric = section['METRIC']

            if 'NORM' in section:
                self.__norm = section['NORM']

            assert self.__channels in [1, 3], 'The number of channels must be 1 or 3'
        except Exception:
            raise ValueError("something wrong with configuration file " + config_path)

    @property
    def model_name(self):
        return self.__modelname

    @property
    def arch(self):
        return self.__arch

    @property
    def process_fun(self):
        return self.__process_fun

    @property
    def number_of_classes(self):
        return self.__number_of_classes

    @property
    def number_of_iterations(self):
        return self.__number_of_iterations

    @property
    def number_of_epochs(self):
        return self.__number_of_epochs

    @property
    def dataset_size(self):
        return self.__dataset_size

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def number_of_batches(self):
        return self.__estimated_number_of_batches

    @property
    def number_of_batches_for_test(self):
        return self.__estimated_number_of_batches_test

    @property
    def save_checkpoints_steps(self):
        return self.__save_checkpoints_steps

    @property
    def test_time(self):
        return self.__test_time

    @property
    def snapshot_dir(self):
        return self.__snapshot_prefix

    @property
    def number_of_channels(self):
        return self.__channels

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def learning_rate(self):
        return self.__lr

    @property
    def lr_decay(self):
        return self.__lr_decay

    @property
    def lr_decay_steps(self):
        return self.__lr_decay_steps

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def class_labels(self):
        return self.__class_labels

    @property
    def deep_feats_layer(self):
        return self.__deep_feats_layer

    @property
    def metric(self):
        return self.__metric

    @property
    def norm(self):
        return self.__norm

    def is_a_valid_section(self, section_name):
        return section_name in self.__config.sections()

    def __str__(self):
        return 'ARCH: {}\n' \
               'NUM_ITERATIONS: {}\n' \
               'DATASET_SIZE: {}\n' \
               'LEARNING_RATE: {}\n' \
               'NUMBER_OF_BATCHES: {}\n' \
               'NUMBER OF EPOCHS: {}\n' \
               'SNAPSHOT_DIR: {}\n' \
               'DATA_DIR: {}'.format(self.arch, self.number_of_iterations, self.dataset_size, self.learning_rate,
                                     self.number_of_batches, self.number_of_epochs, self.snapshot_dir, self.data_dir)
