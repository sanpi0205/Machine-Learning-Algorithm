# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:50:32 2016

@author: zhangbo
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

distances, indices = nbrs.kneighbors(X)
