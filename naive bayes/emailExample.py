# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:00:46 2016

@author: zhangbo
"""

import bayes
import random
import re


text = open("../data/email/ham/6.txt").read()

def textParse(string):
    # 解析文本，将文本输出为词列表
    tokensList = re.split(r'\W*', string)
    return [token.lower() for token in tokensList if len(token) > 2]

def spamTest():
    emailList = []
    classes = []
    
    # 读入数据
    for i in xrange(1,26):
        # 读取垃圾邮件
        email = textParse(open('../data/email/spam/%d.txt' %i).read())
        emailList.append(email)
        classes.append(1)
        # 读取正常邮件
        email = textParse(open('../data/email/ham/%d.txt' %i).read())
        emailList.append(email)
        classes.append(0)
    
    # 构建词向量
    vocabularyList = bayes.vocabularyList(emailList)
    # 构建词频矩阵
    dataSet = bayes.wordsMatrix(vocabularyList, emailList)
    
    # 建立训练集和测试集
    test_num = 20 #测试集数量
    testingData = []
    testingClasses = []
    for i in xrange(test_num):
        testIndex = int(random.uniform(0, len(dataSet)))
        testingData.append(dataSet[testIndex])
        testingClasses.append(classes[testIndex])
        
        # 在原有数据中删除
        del(dataSet[testIndex])
        del(classes[testIndex])
    
    # 训练模型
    p0, p1, pc1 = bayes.trainModel(dataSet, classes)
    
    # 计算测试误差
    errorCount = 0
    i = 0
    for testSample in testingData:
        result = bayes.classifyNB(testSample, p0, p1, pc1)
        if result != testingClasses[i]:
            errorCount += 1
            print "分类错误"
        i += 1
    errorRate = float(errorCount) / test_num
    print "错误率 %f" %errorRate
    return errorRate
        
        
    
    
    
    
        
        
    
    
    
    
    