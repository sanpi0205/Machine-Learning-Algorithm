# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:50:32 2016

@author: zhangbo
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import mnist_test

# 提取数据
mnistData = mnist_test.load_data()

training_inputs = mnistData['training_inputs']
training_labels = mnistData['training_labels']

test_inputs = mnistData['test_inputs']
test_labels = mnistData['test_labels']

neighborModel = KNeighborsClassifier(n_neighbors=7)
neighborModel.fit(training_inputs, training_labels)

zz = neighborModel.predict(test_inputs)

testError = sum(zz == test_labels) / float(len(test_labels))
print "错误率为：%f" %testError

# 0.96


#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
#
#distances, indices = nbrs.kneighbors(X)

