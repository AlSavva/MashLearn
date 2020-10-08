# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:36:09 2020

@author: asavv
"""

# Давайте посмотрим на качество алгоритма в зависимости от количества соседей. 
# Качество будем оценивать на обучающей выборке
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
metrics = []
for n in range(1, 40, 2):
  knn = KNeighborsRegressor(n_neighbors=n)
  scores = cross_val_score(
      knn, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
  metrics.append(np.mean(scores))