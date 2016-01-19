# coding: UTF-8

import numpy as np
import operator

def createDataSet():
    """创建模拟数据"""
    
    group = np.array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])
    return group, labels

def classify0(inX, dataSet, labels, k):
    """利用knn进行分类
    
    参数
    --------
    inX : 待样本点
    
    dataSet: 样本数据
    
    labels: 样本类别（样本标签）
    
    k: 邻居个数，knn算法参数
    
    
    """
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
    # 定义类别计数器，统计k个最近邻的标签
    classCount = {}
    for i in xrange(k):
        label  = labels[sortedDistances[i]]
        classCount[label] = classCount.get(label, 0) + 1
    
    # 对类标签排序,并按照降序排列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse = True)
    
    # 返回排序后第一个类别
    return sortedClassCount[0][0]
    
def knn2(inX, dataSet, labels, k):
    """利用遍历样本方法计算knn"""
    dataSize, dim = dataSet.shape
    
    # 距离向量
    distances = []
    
    for i in xrange(dataSize):
        # 计算欧氏距离
        distance = np.linalg.norm(inX - dataSet[i])
        distances.append(distance)
    
    distances = np.array(distances)
    sortedDistances = distances.argsort()
    classCount = {}
    for i in xrange(k):
        label  = labels[sortedDistances[i]]
        classCount[label] = classCount.get(label, 0) + 1
    
    # 对类标签排序,并按照降序排列
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse = True)
    
    # 返回排序后第一个类别
    return sortedClassCount[0][0]


def file2matrix(filename):
    """读取数据文件"""
    
    fileRead = open(filename)
    
    dataSet = []
    labels = []
    
    # 按行读入文件, 按行读取数据，每次读入一行
    for line in fileRead:
        line = line.strip()
        line = line.split('\t')
        # 将文本数据转换浮点数据
        data = line[0:3]
        data = map(float, data)
        # 将文本分类数据转换为整数
        label = line[-1]
        label = int(label)
        
        dataSet.append(data)
        labels.append(label)
    
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    return dataSet, labels    
    
    
def autoNorm(dataSet):
    """数据归一化处理
    归一化处理公示很多，此处用 (x - min) / (max -min)
    也可以用 (x - mean) / sd
    """

    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    dataRange = maxValues - minValues
    
    # 利用矩阵计算，用空间换时间
    dataSize,dim = dataSet.shape
    normData = np.zeros((dataSize, dim))
    normData = dataSet - np.tile(minValues, (dataSize, 1))
    normData = normData / np.tile(dataRange, (dataSize, 1))
    
    return normData
    
    

        