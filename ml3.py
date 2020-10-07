# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:53:10 2020

@author: asavv
"""
import numpy as np
array = np.array([1,2,4])
print(array * 2)
matrix = np.ones((2,4))
print(matrix)
print(np.sum(matrix, axis = 1))
rand_matrix = np.random.normal(size=(2,2))
print(rand_matrix)
print(matrix.T @ rand_matrix)
print(np.linalg.det(rand_matrix))
print(np.linalg.matrix_rank(rand_matrix))
