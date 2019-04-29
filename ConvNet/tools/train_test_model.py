#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:30:08 2018
@author: jose.saavedra

A convolutional neural network tool 
This implementation are based on using the classes Estimator and Dataset
For more details see cnnLib.cnn

"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cnnLib import cnn
from cnnLib import configuration as conf
from cnnLib import pmapping as pmap
from cnnLib.utils import get_freer_gpu


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training / testing x models")
    parser.add_argument("-mode", type=str, choices=['test', 'train', 'predict', 'save'], help=" test | train ",
                        required=True)
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required=False)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)
    parser.add_argument("-config", type=str, default='config.ini', help=" <optional>, a configuration file", required=False)
    parser.add_argument("-ckpt", type=str,
                        help=" <optional>, it defines the checkpoint for training<fine tuning> or testing",
                        required=False)
    parser.add_argument("-image", type=str, help=" <optional>, a filename for an image to be tested in -predict mode-",
                        required=False)

    pargs = parser.parse_args()

    configuration = conf.ConfigurationFile(pargs.config, pargs.name)
    configuration.show()

    # it is also possible to define the id of the device
    if pargs.device == 'gpu':
        freer_gpu_id = get_freer_gpu()
        device_name = '/device:GPU:{}'.format(freer_gpu_id)
    else:
        device_name = "/cpu:0"
    print('Running on {} device'.format(device_name))

    params = {'device': device_name, 'modelname': pargs.name}

    if pargs.ckpt is not None:
        params['ckpt'] = pargs.ckpt

    my_cnn = cnn.CNN(pargs.config, params)

    run_mode = pargs.mode
    if run_mode == 'train':
        my_cnn.train()
    elif run_mode == 'test':
        my_cnn.test()
    elif run_mode == 'predict':
        print(pargs.image)
        assert os.path.exists(pargs.image), "-image is required"
        prediction = my_cnn.predict(pargs.image)[0]
        probs = prediction['predicted_probabilities']
        idx_class = np.argmax(probs)
        prob = probs[idx_class]
        print("Class: {} [ {} ] ".format(idx_class, prob))
        mapping_file = os.path.join(configuration.data_dir(), "mapping.txt")
        if os.path.exists(mapping_file):
            class_mapping = pmap.PMapping(mapping_file)
            print("Predicted class [{}]".format(class_mapping.getClassName(idx_class)))
        else:
            print("Predicted class [{}]".format(idx_class))
    elif run_mode == 'save':
        my_cnn.save_model()
    print("OK   ")
