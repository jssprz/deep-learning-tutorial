"""
Author: jssprz
"""

import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import mlp.mlp as mlp
import mlp.fast_predictor as fp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model for a specific dataset')
    parser.add_argument('-phase', type=str, help='phase to run: train, test, confusion_matrix, save or predict', required=True)
    parser.add_argument('-ds', type=str, help='name of the dataset to evaluate', required=True)
    input_args = parser.parse_args()
    phase = input_args.phase
    ds = input_args.ds
    assert ds in ['mnist', 'quickdraw'], '{} is not a correct dataset'.format(ds)

    params = {'device': '/cpu:0', 'activation-param': 0.1}

    if ds == 'mnist':
        tmp_model_dir = '../models/MNIST-5000'
        params['data_dir'] = '../../../../datasets/MNIST-5000'
        params['batch_size'] = 200
        params['number_of_iterations'] = 2500  # 100 epochs
        params['data_size'] = 5000
        params['class_labels'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif ds == 'quickdraw':
        tmp_model_dir = '../models/QuickDraw-Animals'
        params['data_dir'] = '../../../../datasets/QuickDraw-Animals'
        params['batch_size'] = 200
        params['number_of_iterations'] = 6000  # 100 epochs
        params['data_size'] = 12000
        params['class_labels'] = ['sheep', 'bear', 'bee', 'cat', 'camel', 'cow', 'crab', 'crocodile', 'duck',
                                  'elephant', 'dog', 'giraffe']

    for act_function in ['sigmoid', 'tanh', 'relu', 'leaky-relu']:
        params['activation'] = act_function
        for opt in ['adam', 'gradient-descent']:
            params['optimizer'] = opt
            for lr in [0.1, 0.01, 0.001]:
                params['learning_rate'] = lr
                params['model_dir'] = os.path.join(tmp_model_dir, '{}/{}/{}'.format(act_function, opt, lr))
                params['save_model_dir'] = os.path.join(params['data_dir'], 'mlp-models/{}/{}/{}'.format(act_function, opt, lr))

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
                elif phase == 'confusion_matrix':
                    print("--------starting computing confunsion matrix for test set")
                    print(my_mlp.confusion_matrix_for_test())
                    print("--------end computing confunsion matrix for test set")
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
