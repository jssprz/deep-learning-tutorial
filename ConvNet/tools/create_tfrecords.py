#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 2018
@author: jsaavedr

Description: A list of function to create tfrecords from a jfile
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cnnLib.data as data
import cnnLib.imgproc as imgproc
import cnnLib.configuration as conf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a dataset for training an testing")
    """ pathname should include train.txt and test.txt, files that should declare the data that will be processed"""
    parser.add_argument("-type", type=int, help="<int> 0: only train, 1: only test, 2: both", required=True)
    parser.add_argument("-config", type=str, help="<str> configuration file", required=True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required=True)
    parser.add_argument("-imheight", type=int, help="<int> size of the image", required=True)
    parser.add_argument("-imwidth", type=int, help="<int> size of the image", required=False)
    pargs = parser.parse_args()
    imh = pargs.imheight
    imw = pargs.imheight
    if not pargs.imwidth is None:
        imw = pargs.imwidth

    configuration_file = pargs.config
    assert os.path.exists(configuration_file), "configuration file does not exist {}".format(configuration_file)
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)
    data.createTFRecord(configuration.data_dir, pargs.type, (imh, imw),
                        number_of_channels=configuration.number_of_channels, processFun=imgproc.process_fun())
    print("tfrecords created for " + configuration.data_dir)
