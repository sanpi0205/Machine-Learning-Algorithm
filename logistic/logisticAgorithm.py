# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:12:37 2016

@author: zhangbo
"""

import numpy as np

def loadData():
    dataMat = []
    labelMat = []
    fr = open('../data/logistic/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 增加三个变量x1, x2, x3 
        # 其中x1是均为1
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        
    return dataMat, labelMat


def sigmoid(z):
    # 连接函数： Link function
    # f(z) = 1 / (1 + e^(-z))
    return 1.0/(1 + np.exp(-z))
    

def costDerivative(y, yhat):
    """sf"""
    return yhat - y

def gradient(dataSet, labels):
    dataSet = np.array(dataSet)
    labels = np.array(labels, ndmin=2).transpose()
    
    n, p = np.shape(dataSet)
    alpha = 0.001
    maxCycles = 1
    # 初始化权重，也可以是随机初始化
    weights = np.ones((p,1))
    for i in xrange(maxCycles):
        yhat = sigmoid( np.dot(dataSet, weights))
        grad = costDerivative(labels, yhat)
        weights += -alpha * np.dot(dataSet.transpose(), grad)
        
        
    return yhat
    