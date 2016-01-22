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
    
    



