# coding: UTF-8


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import knn
import mnist_test

group, labels = knn.createDataSet()
bb = knn.classify0([0,0], group, labels, 3)
print bb
cc = knn.knn2([0,0], group, labels, 3)

# 可视化数据
dataSet, labels = knn.file2matrix('data/datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataSet[:,0], dataSet[:,1], s=15.0*labels, c=15.0*labels)
plt.show()


# 测试误判率
reload(knn)
# testRatio为测试集比例，k为邻居个数
knn.knnTest('../data/datingTestSet2.txt',testRatio=0.2, k=3)


# 测试手写数字识别

mnist_test.mnist_test(500,100, k=7)