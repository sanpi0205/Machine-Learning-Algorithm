# coding: UTF-8

import numpy as np
import operator

def createDataSet():
    """创建模拟数据"""
    
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return group, labels

def classify0(inX, dataSet, labels, k):
    """利用knn进行分类"""
    # 获取数据大小
    dataSize, dim = dataSet.shape
    # 将新数据扩展为与样本一样大小的矩阵，利用矩阵进行计算
    diffMat = np.tile(inX, (dataSize, 1)) - dataSet
    # 将矩阵元素平方
    squareMat = diffMat ** 2
    # 计算距离，沿轴=1求和
    distances = squareMat.sum(axis=1) ** 0.5
    # 计算排序大小，返回下标，利用该下标获取标签，并分类
    sortedDistances = distances.argsort()
    # 定义
    classCount = {}
    
    
    
    
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), 
			reverse=True)
	return sortedClassCount[0][0] 