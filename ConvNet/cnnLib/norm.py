#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:50:16 2018

@author: jsaavedr

Different method for normalization
"""
import numpy as np
from sklearn.preprocessing import normalize


# square root normalization
def square_root_norm(data):
    # square_root
    # add 1 to avoid division by zero
    sign_minus = data < 0
    data_abs = np.abs(data)
    data_sqrt = np.sqrt(data_abs)
    data_sqrt[sign_minus] = data_sqrt[sign_minus] * -1

    if len(data.shape) > 1:
        dim = data.shape[1]
        norm2 = np.sqrt(np.sum(np.square(data_sqrt), axis=1))
        norm2_r = np.repeat(norm2, dim, axis=0)
        norm2_r = np.reshape(norm2_r, [-1, dim])
    else:
        dim = data.size
        norm2 = np.sqrt(np.sum(np.square(data_sqrt)))
        norm2_r = np.repeat(norm2, dim, axis=0)
        norm2_r = np.reshape(norm2_r, [dim])

    normed_data = data_sqrt / norm2_r

    return normed_data


def unit_norm(data):
    if len(data.shape) > 1:
        dim = data.shape[1]
        norm2 = np.sqrt(np.sum(np.square(data), axis=1))
        norm2_r = np.repeat(norm2, dim, axis=0)
        norm2_r = np.reshape(norm2_r, [-1, dim])
    else:
        dim = data.size
        norm2 = np.sqrt(np.sum(np.square(data)))
        norm2_r = np.repeat(norm2, dim, axis=0)
        norm2_r = np.reshape(norm2_r, [dim])
    return data / norm2_r


if __name__ == '__main__':
    # A little test
    a = np.array([[1, 2, 3], [2, -10, 6]])
    print(a)

    nn = np.sqrt(np.sum(a ** 2))
    print(nn)

    b = square_root_norm(a)
    print(b)

    c = unit_norm(a)
    print(c)

    d = normalize(a)
    print(d)
