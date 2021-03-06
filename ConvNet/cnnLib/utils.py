#!/usr/bin/env python
"""Defines some useful functions
"""

import os
import numpy as np

__author__ = "jssprz"
__version__ = "0.0.1"
__maintainer__ = "jssprz"
__email__ = "jperezmartin90@gmail.com"
__status__ = "Development"


# to uint8
def toUINT8(image):
    if image.dtype == np.float64:
        image = image * 255
    elif image.dtype == np.uint16:
        image = image >> 8
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype(np.uint8, copy=False)
    return image


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)