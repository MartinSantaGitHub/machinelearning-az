# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 18:44:55 2020

@author: msantamaria
"""

# Plantilla de Regresión

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values # iloc -> index localization
y = dataset.iloc[:,2].values

# Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
 
# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Ajustar la regresión con el dataset
from sklearn.svm import SVR
regression = SVR(kernel = "rbf",gamma="auto")
regression.fit(X,y.ravel())

# Predicción de nuestros modelos con SVR
x_transformed = sc_X.transform(np.array([[6.5]]))
y_pred = regression.predict(x_transformed)
y_pred_transformed = sc_y.inverse_transform(y_pred)

# Visualización de los resultados del SVR
fig, ax = plt.subplots()
X_t = sc_X.inverse_transform(X)
y_t = sc_y.inverse_transform(y)
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regression.predict(X_grid),color='blue')
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (En dolares)")
#plt.xlim(-2,2)
ax.set_xticklabels(np.round(X_t.ravel(),2))
ax.set_yticklabels(np.round(y_t.ravel(),2))
plt.show()
