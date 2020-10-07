# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:18:24 2020

@author: Алекс Савва
"""
from sklearn.model_selection import train_test_split, GridSearchCV, \
    cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import time
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np 
from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np
t=time.time()
mndata = MNIST('./dir_with_mnist_data_files', gz=True)
images, labels = mndata.load_training()
images, labels = np.array(images), np.array(labels)
plt.imshow(images[0].reshape(28, 28))
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.8, random_state=0)
knn = KNeighborsClassifier(n_neighbors=30)
m=time.time()-t
print(m)
t=time.time()
knn.fit(X_train, y_train)
m=time.time()-t
print(m/60)
t=time.time()
my_predict = knn.predict(X_test)
m=time.time()-t
print(m/60)
t=time.time()
print(accuracy_score(y_test, my_predict))
m=time.time()-t
print(m)
