# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:43:39 2016

@author: zhangbo
"""

import math
#import numpy as np


def createData():
    
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def calculateEntropy(dataSet):

    num = len(dataSet)
    