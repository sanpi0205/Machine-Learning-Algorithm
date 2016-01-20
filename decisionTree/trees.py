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

