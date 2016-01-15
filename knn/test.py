# coding: UTF-8

import knn

group, labels = knn.createDataSet()
bb = knn.classify0([0,0], group, labels, 3)
print bb
