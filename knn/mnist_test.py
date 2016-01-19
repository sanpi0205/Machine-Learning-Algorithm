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
    

def mnist_test(numTrain = 200, numTest = 100, k =5):
    """使用手写数据测试数据"""
    
    training_data, validation_data, test_data = load_data()
    training_inputs = training_data[0][:numTrain]
    training_labels = training_data[1][:numTrain]
    
    test_inputs = test_data[0][:numTest]
    test_lables = test_data[1][:numTest]
    
    
    n = test_inputs.shape[0]
    numError = 0
    for i in xrange(n):
        result = knn.classify0(test_inputs[i], dataSet=training_inputs, 
                               labels=training_labels, k = k)
        if result != test_lables[i]:
            numError += 1
    
    testError = numError / float(n)
    
    print "测试数据错误率为 %f" %testError
    
    
    

