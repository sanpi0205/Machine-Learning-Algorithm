# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:35:10 2016

@author: zhangbo

贝叶斯分类器

"""

import numpy as np
import operator


def loadData():
    """生成数据"""
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def vocabularyList(dataSet):
    """将文本中所有词语汇聚成一个无重复词的List，
    返回值必须是list, 后续要调用list.index
    """
    vocabularySet = set([])
    
    for document in dataSet:
        vocabularySet = vocabularySet | set(document)
    return list(vocabularySet)

def words2vector(vacabularyList, newWords):
    """将新词list，转化为与vacabulary相同长度的0-1数组
    """
    vector = [0] * len(vacabularyList)
    wordsNotIncluded = 0
    for word in newWords:
        if word in vacabularyList:
            vector[vacabularyList.index(word)] = 1
        else:
            print "the word: %s is not in this list" %word
            wordsNotIncluded += 1
    
    if wordsNotIncluded > 0:
        rateNotIncluded = float(wordsNotIncluded)/len(newWords)
        print "共有 %d 个单词不再词来表中，占比: %f" %(wordsNotIncluded, rateNotIncluded)
    
    return vector

def trainModel(dataSet, classes):
    """通过训练数据计算每个词的概率
    并假定这些词是独立的，如果词相互关联怎么处理？
    """
    n = len(dataSet) #数据长度
    numWords = len(dataSet[0]) #列长度，即词列表的长度
    
    #c=1的概率
    p_c1 = np.sum(classes) / float(n)
    # 这样会使输出的概率之和不等于1，但是分类中只涉及两个
    # 样本的比较，因而不会影响最终分类。
    
    # 初始化为1，在比较大小是为单调函数，为使得取对数时不产生错误
    p_w_c0 = np.ones(numWords)
    p_w_c1 = np.ones(numWords)
    
    # 初始化为2，分母不为零，即使所有概率均为零
    p_c0_total = 2.0
    p_c1_total = 2.0
    
    for i in range(n):
        if classes[i] == 1:
            p_w_c1 += dataSet[i]
            p_c1_total += np.sum(dataSet[i])
        else:
            p_w_c0 += dataSet[i]
            p_c0_total += np.sum(dataSet[i])
    
    p1 = np.log( p_w_c1 / p_c1_total)
    p0 = np.log( p_w_c0 / p_c0_total )
    return p0, p1, p_c1

    
    
            