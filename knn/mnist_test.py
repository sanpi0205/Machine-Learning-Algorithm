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

    # 训练数据    
    training_inputs = training_data[0]
    training_labels = training_data[1]
    
    # 验证数据
    validation_inputs = validation_data[0]
    validation_labels = validation_data[1]
    
    # 测试数据
    test_inputs = test_data[0]
    test_labels = test_data[1]
    
    mnistData = {}
    mnistData['training_inputs'] = training_inputs
    mnistData['training_labels'] = training_labels    
    mnistData['validation_inputs'] = validation_inputs
    mnistData['validation_labels'] = validation_labels
    mnistData['test_inputs'] = test_inputs
    mnistData['test_labels'] = test_labels
    
    
    #return training_data, validation_data, test_data
    return mnistData
    

def mnist_test(numTrain = 200, numTest = 100, k =5):
    """使用手写数据测试数据"""
    
    mnistData = load_data()
    training_inputs = mnistData['training_inputs'][:numTrain]
    training_labels = mnistData['training_labels'][:numTrain]
    
    test_inputs = mnistData['test_inputs'][:numTest]
    test_labels = mnistData['test_labels'][:numTest]
    
    
    n = test_inputs.shape[0]
    numError = 0
    for i in xrange(n):
        result = knn.classify0(test_inputs[i], dataSet=training_inputs, 
                               labels=training_labels, k = k)
        if result != test_labels[i]:
            numError += 1
    
    testError = numError / float(n)
    
    print "测试数据错误率为 %f" %testError
    
    
    

