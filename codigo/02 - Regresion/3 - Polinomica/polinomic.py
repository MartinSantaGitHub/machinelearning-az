# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:02:08 2020

@author: msantamaria
"""

# Regresión polinómica

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values # iloc -> index localization
y = dataset.iloc[:,2].values

# Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
# Tenemos muy pocos datos como para dividirlos
 
# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Ajustar la regresión polinómica con todo el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X) 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualización de los resultados del Modelo Lineal
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (En dolares)")
plt.show()

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (En dolares)")
plt.show()

# Predicción de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
