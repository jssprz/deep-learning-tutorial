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
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cnnLib.cnn import CNN
from cnnLib.configuration import ConfigurationFile
from cnnLib.pmapping import PMapping
from cnnLib.deep_searcher import DeepSearcher
from cnnLib.fast_predictor import FastPredictor
from cnnLib.utils import get_freer_gpu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training / testing x models")
    parser.add_argument("-mode", type=str,
                        choices=['test', 'train', 'predict', 'fast_predict', 'deep_searcher', 'conf-mat', 'save'],
                        help=" train | test | predict | fast_predict | deep_searcher | conf-mat | save ", required=True)
    parser.add_argument("-device", type=str, choices=['cpu', 'gpu'], help=" cpu | gpu ", required=False)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)
    parser.add_argument("-config", type=str, default='config.ini', help=" <optional>, a configuration file",
                        required=False)
    parser.add_argument("-ckpt", type=str,
                        help=" <optional>, it defines the checkpoint for training<fine tuning> or testing",
                        required=False)
    parser.add_argument("-list", type=str, help=" <optional>, a list of  images for prediction in -fast_predict mode-",
                        required=False)
    parser.add_argument("-image", type=str,
                        help=" <optional>, a filename for an image to be tested in -predict fast_predict modes-",
                        required=False)

    pargs = parser.parse_args()

    configuration = ConfigurationFile(pargs.config, pargs.name)
    print(configuration)

    # it is also possible to define the id of the device
    if pargs.device == 'gpu':
        freer_gpu_id = get_freer_gpu()
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(freer_gpu_id)
        device_name = '/device:GPU:{}'.format(freer_gpu_id)
    else:
        device_name = "/cpu:0"
    print('Running on {} device'.format(device_name))

    params = {'device': device_name, 'modelname': pargs.name}

    if pargs.ckpt is not None:
        params['ckpt'] = pargs.ckpt

    my_cnn = CNN(pargs.config, params)

    run_mode = pargs.mode
    if run_mode == 'train':
        my_cnn.train()
    elif run_mode == 'test':
        my_cnn.test()
    elif run_mode == 'conf-mat':
        my_cnn.confusion_matrix_for_test()
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
            class_mapping = PMapping(mapping_file)
            print("Predicted class [{}]".format(class_mapping.get_class_name(idx_class)))
        else:
            print("Predicted class [{}]".format(idx_class))
    elif run_mode == 'fast_predict':
        faster_predictor = FastPredictor(configuration, params)
        if pargs.list is not None:
            with open(pargs.list) as f_in:
                list_images = [item.strip() for item in f_in]
            avg = 0
            for item in list_images:
                start = time.time()
                predicted_probs, predicted_classes = faster_predictor.predict(item)
                print("{} -> {}".format(predicted_classes[0], predicted_probs[0]))
                print("OK")
                end = time.time()
                avg = avg + end - start
            print("Average elapsed time {} ".format(avg / len(list_images)))
        elif pargs.image is not None:
            filename = pargs.image
            while True:
                start = time.time()
                predicted_probs, predicted_classes = faster_predictor.predict(filename)
                print("{} -> {}".format(predicted_classes[0], predicted_probs[0]))
                end = time.time()
                print("Elapsed time {} ".format(end - start))
                filename = input("Image: ")
                while len(filename) == 0:
                    filename = input("Image: ")
    elif run_mode == 'deep_search':
        deep_searcher = DeepSearcher(configuration, params)
        t_start = time.time()
        print(deep_searcher.mean_average_precision(deep_searcher.feats_vectors, deep_searcher.true_labels))
        t_elapsed = (time.time() - t_start) * 1000
        print(">>> computes the mean_average_precision took {} ms".format(t_elapsed))
    elif run_mode == 'save':
        my_cnn.save_model()
    print("OK   ")
