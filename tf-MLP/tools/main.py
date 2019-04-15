#!/usr/bin/env python
"""Provides the main function
The tests can be executed from command line setting the dataset and the phase
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.mlp as mlp
import mlp.fast_predictor as fp

__author__ = "jssprz"
__version__ = "0.0.2"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"

datasets = ['mnist', 'quickdraw']
act_functions = ['sigmoid', 'tanh', 'relu', 'leaky-relu']
optimizers = ['adam', 'gd']
lrs = [0.01, 0.001]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model for a specific dataset')
    parser.add_argument('-phase', type=str, help='phase to run: train, test, confusion_matrix, save or predict', required=True)
    parser.add_argument('-ds', type=str, help='name of the dataset to evaluate', required=True)
    parser.add_argument('-lr', type=float, help='learning rate parameter to be used', required=False)
    parser.add_argument('-af', type=str, help='name of the activation function to be used', required=False)
    parser.add_argument('-opt', type=str, help='name of the optimizer method to be used', required=False)
    parser.add_argument('-chkpt', type=int, help='iteration number of the checkpoint to be loaded', required=False)
    input_args = parser.parse_args()
    phase = input_args.phase
    ds = input_args.ds
    lr = input_args.lr
    act_function = input_args.af
    optimizer = input_args.opt
    checkpoint = input_args.chkpt

    assert ds in datasets, '{} is not a correct dataset'.format(ds)

    params = {'device': '/cpu:0', 'save_checkpoints_steps': 500, 'activation_param': 0.1}

    if ds == 'mnist':
        tmp_model_dir = '../models/MNIST-5000'
        params['data_dir'] = '../../../../datasets/MNIST-5000'
        params['batch_size'] = 50
        params['number_of_iterations'] = 27000  # 270 epochs
        params['data_size'] = 5000
        params['class_labels'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:  # QuickDraw
        tmp_model_dir = '../models/qda-tmp1'
        params['data_dir'] = '../../../../datasets/QuickDraw-Animals'
        params['batch_size'] = 80
        params['number_of_iterations'] = 40000  # 270 epochs
        params['data_size'] = 12000
        params['class_labels'] = ['sheep', 'bear', 'bee', 'cat', 'camel', 'cow', 'crab', 'crocodile', 'duck',
                                  'elephant', 'dog', 'giraffe']

    if checkpoint is not None:
        assert optimizer in optimizers, '{} is not a correct optimizer method'.format(optimizer)
        assert act_function in act_functions, '{} is not a correct activation function'.format(act_function)
        assert lr in lrs, '{} is not a correct learning rate value'.format(lr)

        params['activation'] = act_function
        params['optimizer'] = optimizer
        params['learning_rate'] = lr
        params['model_dir'] = os.path.join(tmp_model_dir, '{}/{}/{}'.format(act_function, optimizer, lr))
        params['save_model_dir'] = os.path.join(params['data_dir'],
                                                'mlp-models/{}/{}/{}'.format(act_function, optimizer, lr))

        my_mlp = mlp.MLP(params)
        print("MLP initialized ok")

        if phase == 'confusion_matrix':
            print("--------starting computing confusion matrix for test set")
            print(my_mlp.confusion_matrix_for_test(checkpoint))
            print("--------end computing confusion matrix for test set")
        elif phase == 'save':
            print("--------start saving trained model")
            my_mlp.save_model(checkpoint)
            print("--------end saving trained model")
    else:
        for act_function in act_functions:
            params['activation'] = act_function
            for opt in optimizers:
                params['optimizer'] = opt
                for lr in lrs:
                    params['learning_rate'] = lr
                    params['model_dir'] = os.path.join(tmp_model_dir, '{}/{}/{}'.format(act_function, opt, lr))
                    params['save_model_dir'] = os.path.join(params['data_dir'], 'mlp-models/{}/{}/{}'.format(act_function, opt, lr))

                    if not os.path.exists(params['save_model_dir']):
                        os.makedirs(params['save_model_dir'])

                    my_mlp = mlp.MLP(params)
                    print("MLP initialized ok")

                    if phase == 'train':
                        print("--------start training")
                        my_mlp.train()
                        print("--------end training")
                    elif phase == 'test':
                        print("--------start testing")
                        result = my_mlp.test()
                        print(result)
                        print("--------end testing")
                    if phase == 'confusion_matrix':
                        print("--------starting computing confusion matrices for test set")
                        print(my_mlp.confusion_matrix_for_test())
                        print("--------end computing confusion matrices for test set")
                    elif phase == 'save':
                        print("--------start saving trained model")
                        my_mlp.save_model()
                        print("--------end saving trained model")
                    elif phase == 'predict':
                        print("--------start infinite predictions")
                        predictor = fp.FastPredictor(params)
                        while True:
                            filename = input("Image: ")
                            prediction = predictor.predict(filename)
                            print(prediction)
