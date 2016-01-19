# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:57:26 2016

@author: zhangbo
"""
import numpy as np
import cPickle
import gzip

import knn


def load_data():
    """加载MNIST数据集"""
    
    f = gzip.open('../data/mnist.pkl.gz', 'r')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    
    return training_data, validation_data, test_data
    
