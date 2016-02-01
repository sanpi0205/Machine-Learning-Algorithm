# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:12:37 2016

@author: zhangbo
"""

import numpy as np
import matplotlib.pyplot as plt

def loadData():
    dataSet = []
    labels = []
    fr = open('../data/logistic/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 增加三个变量x1, x2, x3 
        # 其中x1是均为1
        dataSet.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labels.append(int(lineArr[2]))
        
    dataSet = np.array(dataSet)
    n = dataSet.shape[0]
    labels = np.array(labels).reshape((n, 1))
    return dataSet, labels


def sigmoid(z):
    # 连接函数： Link function
    # f(z) = 1 / (1 + e^(-z))
    return 1.0/(1 + np.exp(-z))
    
def cost(y, yhat):
    """cross entropy损失函数：c = -1/n * sum( y*ln(yhat) + (1-y)*log(1-yhat) )
    """
    n = y.shape[0]
    costValue = -(1.0/n) * np.sum( y*np.log(yhat) + (1-y)*np.log(1-yhat) )
    return costValue

def costDerivative(y, yhat):
    """"""
    return (yhat - y)

def gradient(dataSet, y):
    
    n, p = np.shape(dataSet)
    alpha = 0.003
    maxCycles = 300
    # 初始化权重，也可以是随机初始化    
    weights = np.ones((p,1))

    # 记录损失函数与权重之和    
    costValues = []
    weightsSum = []
    for i in xrange(maxCycles):
        yhat = sigmoid( np.dot(dataSet, weights))
        costValues.append(cost(y, yhat))
        grad = costDerivative(y, yhat)
        weights += -alpha * np.dot(dataSet.transpose(), grad)
        weightsSum.append(np.sum( np.abs(weights)))
        
    return costValues, weightsSum, weights

def plotFitLine(weights):
    dataSet, labels = loadData()
    n, p = np.shape(dataSet)
    x1 = []; y1 = []
    x2 = []; y2 = []
    for i in xrange(n):
        if labels[i] == 1:
            x1.append(dataSet[i][1])
            y1.append(dataSet[i][2])
        else:
            x2.append(dataSet[i][1])
            y2.append(dataSet[i][2])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='blue')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.show()
    
