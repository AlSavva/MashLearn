# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:25:01 2020

@author: asavv
"""

# Введение в Pandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Будем работать с данными, собранными благодаря опросу студентов 
#математического курса средней школы в Португалии (возраст - от 15 до 22 лет).
#Они находятся в файле "math_students.csv". Целевой переменной является 
#итоговая оценка студента за курс.
data = pd.read_csv('math_students/math_students.csv', delimiter=',')
print(data.head())# функция .head(n) выводит первые n строк таблицы (по умолчанию n=5)
print(data.shape)#  Method .shape возвращает кортеж (кол-во строк, кол-во столбцов базы) 
print(data.tail())# функция .tail(n) выводит первые n строк таблицы (по умолчанию n=5)
print(data.columns)# выводит названия столбцов базы
# Строки 17-20 выадют базу без последнего столбца
print(data[data.columns[:-1]].head())
print(data.iloc[:, :-1].head())#Указываются индексы строк и столбцов
print(data.loc[:, data.columns[:-1]].head())#Указываются имена строк и столбцов
print(data.drop(['G3'], axis=1).head())#параметр axis=1 показываетб что 
#исключается столбец. по умолчанию=0 - строка
print(data.drop([4]).head())
print(data.isnull().any().any())#показывает есть ли в данных пропуски 
#.any(axis=0 - default - columns)
print(data.describe())#статистика по значениям признаков
my_dat_descr = data.describe()
data.info()#Более подробное описание значений признаков (количество 
#непустых значений, типов столбцов и объема занимаемой памяти)без print!
print(data['guardian'].unique())#Какие значения принимает признак
print(my_dat_descr['Walc'].unique())
print(data['guardian'].nunique())#Количество значений которые принимает признак
print(my_dat_descr['Walc'].nunique())
print(data['guardian'].value_counts())#Количество повторений значений которые 
#принимает признак
print(my_dat_descr['Walc'].value_counts())
#Выделим только тех студентов, у которых опекуном является мать и которая 
#работает учителем или на дому
my_filter_data = data[(data['guardian'] == 'mother') 
                      & ((data['Mjob'] == 'teacher') 
                         | (data['Mjob'] == 'at_home'))]
print(my_filter_data)
print(my_filter_data.shape)
#Помимо имеющихся признаков, можно создавать новые, которые могут оказаться 
#полезными для построения качественного алгоритма. Например, внимательно 
#изучив описания признаков, связанных с алкоголем, создадим признак "alc", 
#который будет отражать общее употребление алкоголя в течение недели по 
#формуле alc = (5 * Dalc + 2 * Walc) / 7
data['alc'] = (5 * data['Dalc'] + 2 * data['Walc']) / 7#Добавляем признак в базу
print(data[['Walc', 'Dalc', 'alc']])#выводим требуемые столбцы
data_alc = data[['Walc', 'Dalc', 'alc']]
data_alc_descr = data_alc.describe()
print(data.columns)
#Проанализируем взаимосвязь количества пропусков и успехов в учебе. Посмотрим 
#на распределение количества пропусков у студентов:
plt.figure(figsize=(10,7))#size of picture
plt.title('Absences distribution')#graf name
data['absences'].hist()#graf type - histigramma
plt.xlabel('absences')# name of x-line
plt.ylabel('number of students')#----y-line
plt.show()#?
#Посмотрим на среднее количество пропусков:
mean_absences = data['absences'].mean()
print(mean_absences)
#из гистограммы и полученного значения можно сделать вывод, что большинство 
#студентов пропускает не так много занятий. Теперь посмотрим на влияние 
#количества пропусков на итоговую оценку. Для этого разделим студентов на две 
#части: те, у кого количество пропусков меньше среднего, и те, у кого оно не 
#меньше среднего.
stud_few_absences = data[data['absences'] < mean_absences]
stud_many_absences = data[data['absences'] >= mean_absences]
print('Students with few absences:', stud_few_absences.shape[0])
print('Students with many absences:', stud_many_absences.shape[0])
#Посчитаем среднее значение целевой переменной("G3") для каждой части.
stud_few_absences_g3 = stud_few_absences['G3'].mean()
stud_many_absences_g3 = stud_many_absences['G3'].mean()
print('Students with few absences, mean G3:', stud_few_absences_g3)
print('Students with many absences, mean G3:', stud_many_absences_g3)
#Итак, средние оценки примерно одинаковы - у тех, кто пропускал меньше 
#занятий, она чуть хуже. Возможно, студенты, пропускавшие много занятий, 
#знали материал очень хорошо :) Впрочем, подобное исследование не позволяет 
#делать никаких серьезных выводов.
#Также данные можно исследовать с помощью группировки и агрегирования. 
#Например, найдем исследуем закономерности, связанные с разными школами. 
#Сгруппируем объекты по школам:
data_by_school = data.groupby('school')
print(data_by_school.describe())
my_dabs = data_by_school.aggregate(np.mean)#вернет объект DataFrame 

#Или по статусу отношений родителейЖ
data_by_pstatus = data.groupby('Pstatus')
my_dabp = data_by_pstatus.aggregate(np.mean)
print(data_by_pstatus.describe())
#Теперь посмотрим на среднее значение признаков для каждой школы:
print(data_by_school.mean())
print(data_by_pstatus.mean())
#Можно заметить, например, что в среднем до школы Mousinho da Silveira 
#студентам нужно добираться дольше, чем до Gabriel Pereira. Интересно, 
#что, несмотря на это, в среднем количество пропусков у них меньше.
    


