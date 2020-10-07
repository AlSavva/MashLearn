# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:36:01 2020

@author: asavv
"""

import pandas as pd
import datetime
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
df = pd.read_csv(
    'nyc-taxi-trip-duration/train.zip', compression='zip', header=0, sep=',', quotechar='"')
print(df.head())

#Удалим колонку, которая есть только в обучающей выборке dropoff_datetime. 
#Из названия понятно, что используя эту колонку и pickup_datetime мы сможем 
#восстановить длину поездки. Очевидно, что в начале поездки dropoff_datetime 
#нам недоступна, а значит и для предсказания ее использовать нельзя.
df = df.drop('dropoff_datetime', axis=1)

#Сейчас даты записаны в виде строк. Давайте преобразуем их в питонячие 
#datetime объекты. Таким образом мы сможем выполнять арифметические операции 
#с датами и вытаскивать нужную информацию, не работая со строками.
df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

#Давайте разобьем выборку на train и test. Применить функцию train_test_split 
#в этот раз не получиться. Мы теперь имеем дело с временными данными и на 
#практике наша модель должна уметь работать во временных периодах, которых 
#нет в обучающей выборке. Поэтому разбивать мы будем датасет по хронологии. 
#Для этого отсортируем датасет по дате.
df = df.sort_values(by='pickup_datetime')
df_train = df[:10 ** 6]
df_test = df[10 ** 6:]

#мы будем пресказывать переменную trip_duration. Посмотрим на target переменную.

#!df_train.trip_duration.hist(bins=100, grid=False, )

#Что то пошло не так. Вероятно, есть очень длинные поездки и короткие. 
#Попробуем взять log(1 + x) от длины поездки. Единицу мы прибавляем, чтобы 
#избежать проблем с поездками, которые например мнгновенно завершились.
np.log1p(df_train.trip_duration).hist(bins=100, grid=False, )

#Мы получили куда более ясную картину, распределение стало похоже на 
#нормальное. Работать будем теперь с логарифмом. Так линейной регрессии будет 
#куда проще выучить корректную зависимость. А если захотим вернуться к 
#исходным данным, возведем предсказание в экспоненту.
df_train['log_trip_duration'] = np.log1p(df_train.trip_duration)
df_test['log_trip_duration'] = np.log1p(df_test.trip_duration)

df.pickup_datetime = pd.to_datetime(df.pickup_datetime)

#Посмотрим на наши признаки. Давайте нарисуем, как выглядит распределение 
#количества поездок по дням.
date_sorted = df_train.pickup_datetime.apply(lambda x: x.date()).sort_values()

plt.figure(figsize=(25, 5))
date_count_plot = sns.countplot(
  x=date_sorted,
)
date_count_plot.set_xticklabels(date_count_plot.get_xticklabels(), rotation=90)

#Мы можем увидеть паттерны, которые повторяются каждую неделю. Также мы можем 
#наблюдать несколько аномальных правалов в количестве поездок. Посмотрим, как 
#выглядит распределение по часам.
sns.countplot(df_train.pickup_datetime.apply(lambda x: x.hour), )

#Теперь давайте посмотрим, как связан день и длина поездки.
group_by_weekday = df_train.groupby(df_train.pickup_datetime.apply(
    lambda x: x.date()))
sns.relplot(data=group_by_weekday.log_trip_duration.aggregate(
    'mean'), kind='line')

#Мы видим явный тренд. Более того, наблюдается такая вещь как сезонность: 
#повторяющиеся временные паттерны. В нашем случае период равен неделе.

#Теперь подготовим датасет. Включим в него день года и час дня. Для этого 
#напишем функцию create_features, которая будет собирать нам нужные признаки 
#в отдельный pandas.DataFrame. В итоге, мы сможем воспользоваться этой 
#функцией, как для train подвыборки, так и для test.

def create_features(data_frame):
  X = pd.concat([
      data_frame.pickup_datetime.apply(lambda x: x.timetuple().tm_yday),
      data_frame.pickup_datetime.apply(lambda x: x.hour),
     ], axis=1, keys=['day', 'hour',]
  )
  
  return X, data_frame.log_trip_duration
X_train, y_train = create_features(df_train)
X_test, y_test = create_features(df_test)
#Переменная час, хоть и является целым числом, не может трактоваться как 
#вещественная. Дело в том, что после 23 идет 0, и что будет означать 
#коэффициент регрессии в таком случае, совсем не ясно. Поэтому применим к 
#этой переменной one -hot кодирование. В тоже время, переменная день должна 
#остаться вещественной, так как значения из обучающей выборке не встреться 
#нам на тестовом подмножестве.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([("One hot", OneHotEncoder(sparse=False),[1])], remainder="passthrough")
X_train = ohe.fit_transform(X_train)
X_test = ohe.transform(X_test)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
#Воспользуемся классом Ridge и обучим модель
from time import time
k = time()
ridge = Ridge(alpha=1000).fit(X_train, y_train)
print((time() - k)/60)
print(mean_squared_error(ridge.predict(X_test), y_test))\

#Давайте попробуем сделать лучше и подберем гиперпараметры модели.
from sklearn.model_selection import GridSearchCV

grid_searcher = GridSearchCV(Ridge(),
                             param_grid={'alpha': np.linspace(100, 750, 10)},
                             cv=5).fit(X_train, y_train)
print(time())
print(mean_squared_error(grid_searcher.predict(X_test), y_test))
print(grid_searcher.best_params_)
print(time())


