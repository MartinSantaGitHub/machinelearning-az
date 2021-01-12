# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 07:24:02 2020

@author: msantamaria
"""

# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values # iloc -> index localization
y = dataset.iloc[:,-1].values

# Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state = 0)
 
# Escalado de variables
# No es necesario en la regresión lineal simple

# Crear modelo de regresión lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

# Predecir el conjunto de test
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regression.predict(X_train),color='blue')
plt.title("Sueldo vs Años de Experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueld (En dolares)")
plt.show()

# Visualizar los resultados de test
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regression.predict(X_train),color='blue')
plt.title("Sueldo vs Años de Experiencia (Conjunto de testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueld (En dolares)")
plt.show()
