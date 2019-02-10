# -*- coding: utf-8 -*-
"""
@author: ASUS
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import toimage
plt.rcParams['figure.figsize'] = (14, 8)



def rgb2gray(img):
    w = np.array([0.2989, 0.5870, 0.1140]).reshape((1, 1, -1))
    return (img * w).sum(axis=2)

def rgba2gray(img):
    w = np.array([0.2989, 0.5870, 0.1140, 0.0]).reshape((1, 1, -1))
    return (img * w).sum(axis=2)

def imshow(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        if img.dtype == np.dtype('int'):
            f = plt.imshow(img, cmap='gray', clim=(0, 255))
        else:
            f = plt.imshow(img, cmap='gray')
    else:
        f = plt.imshow(img)
    plt.show()
    return f
        

def imwrite(filename, matrix):
    toimage(matrix, cmin=0, cmax=256).save(filename)
 

