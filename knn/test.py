# coding: UTF-8

import knn
import matplotlib
import matplotlib.pyplot as plt

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

