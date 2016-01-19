# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:05:41 2016

@author: zhangbo
"""

import numpy as np


def euclideanDistance(x, y):
    return 0.5 * np.linalg.norm( x-y )**2

def initCentroids(data, k):
    numSample, dim = data.shape
    centroids = np.zeros(k, dim)
    