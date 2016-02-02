# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:53:10 2016

@author: zhangbo
"""

import numpy as np

def loadData():
    dataSet = np.array([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    
    return dataSet, labels

def stumpClassifier(dataMat, dimension, thresholdValue, threshold):
    result = np.ones((np.shape(dataMat)[0], 1))
    if threshold == 'lt':
        result[dataMat[:, dimension] <= thresholdValue] = -1.0
    else:
        result[dataMat[:, dimension] > thresholdValue] = -1.0
    return result

def buildStump(dataSet, labels, D):

    dataMat = np.mat(dataSet)
    labelsMat = np.mat(labels).T
    
    
    