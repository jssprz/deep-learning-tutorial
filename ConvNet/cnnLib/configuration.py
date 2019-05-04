#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:00:28 2018

@author: jsaavedr
"""

import json
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
            if 'PROCESS_FUN' in section is not None:
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
                self.__lr_decay_steps = float(section['DECAY_STEPS'])

            self.__estimated_number_of_batches = int(float(self.__dataset_size) / float(self.__batch_size))
            self.__estimated_number_of_batches_test = int(float(self.__test_size) / float(self.__batch_size))
            self.__number_of_epochs = int(float(self.__number_of_iterations) / float(self.__estimated_number_of_batches))
            self.__channels = int(section['CHANNELS'])
            self.__class_labels = section['CLASS_LABELS'].split(',')

            assert (self.__channels == 1 or self.__channels == 3), 'The number of channels must be 1 or 3'
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

    def is_a_valid_section(self, section_name):
        return section_name in self.__config.sections()

    def show(self):
        print("ARCH: {}".format(self.arch))
        print("NUM_ITERATIONS: {}".format(self.number_of_iterations))
        print("DATASET_SIZE: {}".format(self.dataset_size))
        print("LEARNING_RATE: {}".format(self.learning_rate))
        print("NUMBER_OF_BATCHES: {}".format(self.number_of_batches))
        print("NUMBER OF EPOCHS: {}".format(self.__number_of_epochs))
        print("SNAPSHOT_DIR: {}".format(self.snapshot_dir))
        print("DATA_DIR: {}".format(self.data_dir))
