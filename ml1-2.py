# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:24:11 2020

@author: asavv
"""

# Возьмем стандартный датасет c помощью функции load_boston. Датасет содержит 
# информацию о ценах на квартиры и какие-то параметры квартиры: близость к реке, 
# криминогенная обстановка в районе и т.д.
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
data = load_boston()
print(data['DESCR'])# печатаем описание базы
print(data['feature_names'])# печатаем названия столбцов
#Значения признаков находятся по ключу"data" значения целевой переменной - цены
# "target"
X, y = data['data'], data['target']
print("Размер матрицы объектов: ", X.shape)
print("Рaзмер вектора y: ", y.shape)
# Посмотрим на наши данные. Давайте построим график того, как зависит цена от 
# криминогенной обстановки. Для этого воспользуемся библиотекой matplotlib.
plt.figure(figsize=(10,7))
plt.xlabel('Crime rate')
plt.ylabel('Price')
plt.scatter(X[:, 0], y)
#Как мы видим, все закономерно, дорогие квартиры находятся в районах с низким 
#уровнем преступности
# С помощью функции train_test_split разобьем выборку на train и test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# В sklearn, разные методы лежат в разных модулях. Например в linear_model 
# находятся линейные модели, в neighbors - методы основанные на ближайших 
# соседях.
from sklearn.neighbors import KNeighborsRegressor
# Импортируем алгоритм knn из sklearn. Работа с алгоритмами машинного обучения 
# в библиотеке состоит из трех этапов.

# Создание объекта, который будет реализовывать алгоритм.
# Вызов fit: обучение модели на тренировочной подвыборке
# Вызов predict: получение предсказаний на тестовой выборке
knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', p=2)#p - указывает 
# на метрику близости. р = 2 - евклидово расстояниеб р = 1 - манхэттенское
# KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
#                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
#                     weights='uniform')
knn.fit(X_train, y_train)# обучаем модель
print(knn.get_params())
predictions = knn.predict(X_test)# получаем предсказания
# Посчитаем метрику, соответствующая функция есть в scikit-learn! Будет считать 
# средне квадратичную ошибку, так как мы решаем задачу регрессии.
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, predictions))
# Давайте попробуем сделать лучше! У нашего алгоритма есть множество 
# гиперпараметров: количество соседей, параметры метрики и веса. Запустим поиск 
# по сетке гиперараметров, алгоритм переберет все возможные комбинации, 
# посчитает метрику для каждого набора и выдаст лучший набор.
from sklearn.model_selection import GridSearchCV
grid_searcher = GridSearchCV(KNeighborsRegressor(),
                             param_grid={'n_neighbors': range(1, 40, 2),
                                         'weights': ['uniform', 'distance'],
                                         'p': [1, 2, 3]},
                             cv=5)
# Параметр cv=5 говорит, что во время поиска оптимальных парамертов будет 
# использоваться кросс-валидация с 5 фолдами.
grid_searcher.fit(X_train, y_train)# обучаем
print(grid_searcher.best_params_)#смотрим лучшие параметры для обучения
# Попросим предсказание лучшей модели
best_predictions = grid_searcher.predict(X_test)
print(mean_squared_error(y_test, best_predictions))
# Заметно что MSE уменьшилась на 30%
# Давайте посмотрим на качество алгоритма в зависимости от количества соседей. 
# Качество будем оценивать на обучающей выборке
from sklearn.model_selection import cross_val_score
# В цикле будем перебирать количество соседей от 1 до 30 с шагом 3, и 
# результаты запишем в метрику
metrics = []
for n in range(1, 40, 2):
  knn = KNeighborsRegressor(n_neighbors=n)
  knn.fit(X_train, y_train)
  metrics.append(mean_squared_error(y_test, knn.predict(X_test)))
# и нарисуем график зависимости MSE от количества соседей:
plt.figure(figsize=(10,7))
plt.plot(range(1, 40, 2), metrics)


