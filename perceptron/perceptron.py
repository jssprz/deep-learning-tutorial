'''
Created on Feb 28, 2019

@author: jsaavedr

A simple implementation of a logistic regression (Perceptron)
'''
import numpy as np

def logsig(x):
    """
    sigmoide function
    """
    return 1.0 / ( 1.0 + np.exp( - x ))


def train(data, target, number_of_iterations, lr=0.01):
    """
    training a simple perceptron using  the gradient descent algorithm
    """    
    n, d = data.shape
    assert n == target.size, "target size incompatible with data size"
    
    ones = np.ones((n,1))
    data = np.append(ones, data, axis = 1)
    d = d + 1    
    w = np.random.rand(d)
    for it in range(number_of_iterations) :
        v = np.matmul(data,w)
        y = logsig(v)        
        #using mse loss
        #loss = np.mean(0.5 * ((target - y) ** 2))      
        #dif =  np.mean((y - target) * (y * (1.0 - y)) * np.transpose(data) , axis = 1)
        #using Cross-Entropy loss        
        loss = - np.mean((target * np.log(y) + (1 - target) * np.log(1 - y)))
        dif =   np.mean( (y - target) * np.transpose(data) , axis = 1)
        w = w - lr * dif
        if it % 100 == 0 :          
            print("loss ({}): {}".format(it, loss))
    return w
    
def predict(data, w): 
    n = data.shape[0]   
    ones = np.ones((n,1))
    data = np.append(ones, data, axis = 1)
    return logsig(np.matmul(data, w))
    
    
