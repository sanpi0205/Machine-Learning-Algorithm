# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:12:37 2016

@author: zhangbo
"""

import numpy as np
import matplotlib.pyplot as plt
import random

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

def costDerivative(dataSet, y, yhat):
    """严格按照函数定义，损失函数的导数应该是 grad  = (1/n) * x^T * (yhat - y)
    """
    n = dataSet.shape[0]
    grad = np.dot(dataSet.transpose(), yhat-y) / n
    return grad

def gradient(dataSet, y):
    
    n, p = np.shape(dataSet)
    alpha = 0.3
    maxCycles = 300
    # 初始化权重，也可以是随机初始化    
    weights = np.ones((p,1))

    # 记录损失函数与权重之和    
    costValues = []
    weightsSum = []
    for i in xrange(maxCycles):
        yhat = sigmoid( np.dot(dataSet, weights))
        costValues.append(cost(y, yhat))
        grad = costDerivative(dataSet, y, yhat)
        weights += -alpha *  grad
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
    
def SGD(dataSet, y):
    """随机梯度算法与梯度下降算法的不同在于，随机梯度算法每次只利用一个样本求解梯度，
    之后将该梯度用于迭代，而梯度下降算法每次迭代需要利用所有数据，算法的计算开销比较大，
    特别是当数据量比较大时，随机梯度算法效率更高。
    当然随机梯度算法，也可以每次利用有限样本求接梯度，成为 mini_batch 的随机梯度算法，
    算法具体代码参加：[Neural Network and Deep Learning]
    """
    n, p = dataSet.shape
    alpha = 0.01
    weights = np.ones(p)
    #for j in xrange(200):
    #    randomIndex = np.random.choice(range(n),size=n,replace=False)
    for i in xrange(n):
        #alpha = 4 / (1.0 + i + j) + 0.01
        h = sigmoid(np.sum(dataSet[i] * weights))
        error = h - y[i]
        weights += -alpha * dataSet[i] * error
    return weights

def SGDmodify(dataSet, y, numIter = 150):
    
    n, p = np.shape(dataSet)
    weights = np.ones(p)
    
    for j in xrange(numIter):
        """原书中的代码，这里做一些优化，优化思路从 1..n 中无放回抽样，即可
        得到 1..n 的随机序列
        
        dataIndex = range(n)
        for i in xrange(n):
            alpha = 4 / (1.0 + i + j) + 0.01
            randomNum = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataSet[dataIndex[randomNum]] * weights))
            error = h - y[dataIndex[randomNum]]
            weights += -alpha * dataSet[dataIndex[randomNum]] * error
            del(dataIndex[randomNum])
        """
        randomIndex = np.random.choice(range(n),size=n,replace=False)
        for i in xrange(n):
            alpha = 4 / (1.0 + i + j) + 0.01
            h = sigmoid(np.sum(dataSet[randomIndex[i]] * weights))
            error = h - y[randomIndex[i]]
            weights += -alpha * dataSet[randomIndex[i]] * error
        
    return weights
    
    
    
    

        