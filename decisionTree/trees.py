# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:43:39 2016

@author: zhangbo
"""

import math
import operator
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
    labelCounts = {}
    
    for row in dataSet:
        label = row[-1]
        #print label
        if label not in labelCounts.keys():
            labelCounts[label] = 1
        else:
            labelCounts[label] += 1
        
    entropy = 0.0
    for key in labelCounts:
        probability = float(labelCounts[key]) / num
        entropy -= probability * math.log(probability, 2)
        
    return entropy
    
def spliteDataSet(dataSet, axis, value):
    """在特定变量和特定点上划分数据，
    划分后的数据将比原来数据集少一个维度，即减少了划分变量，
    也就是说在二分类树的情况下，每次划分都有且只有一个变量将数据集划分两个部分
    """
    
    splitedData = []
    for row in dataSet:
        if row[axis] == value:
            leftDataSet = row[:axis]
            rightDataSet = row[axis+1:]
            # 拼接数据
            leftDataSet.extend(rightDataSet)
            splitedData.append(leftDataSet)
    return splitedData

def chooseBestFeature(dataSet):
    """选择最优划分的变量（特征）"""
    
    numFeature = len(dataSet[0]) - 1 #dataSet中最后一列是y
    #num = len(dataSet) # 样本点个数
    baseEntropy = calculateEntropy(dataSet)
    bestInfoGain = 0.0 #计算信息增益或熵
    bestFeature = -1
    
    for i in xrange(numFeature):
        featureData = [row[i] for row in dataSet]
        uniqueValues = set(featureData)
        
        entropy = 0.0
        for value in uniqueValues:
            splitedData = spliteDataSet(dataSet, i, value)
            #probability = len(splitedData) / float(num)
            entropy +=  calculateEntropy(splitedData)
        
        infoGain = baseEntropy - entropy # 获取最大信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCount(classList):
    """在决策树的节点中，投票预测新节点，并返回最大的类别"""
    
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems, key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def createTree(dataSet, labels):
    """创建决策树"""
    
    # 获取当前数据中不同类，如果使用pandas?
    classList = [row[-1] for row in dataSet]
    # 如果dataSet中只有一个类则停止递归，返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果dataSet中只有一个自变量，则无法继续划分数据，也停止递归
    # 此时返回类别中频次最大的类别
    if len(dataSet[0]) == 1:
        return majorityCount(classList)
    bestFeature = chooseBestFeature(dataSet)
    bestLabels = labels[bestFeature]
    
    
    