# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:40:08 2021

@author: msantamaria
"""

# Grid Search

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values # iloc -> index localization
y = dataset.iloc[:,4].values

# Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state = 0)
 
 # Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", random_state=0)
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)

# Aplicar k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Aplicar la mejora de Grid Search para optimizar el modelo y sus parámetros
from sklearn.model_selection import GridSearchCV
parameters = [{"C": [1,10,100,1000], 
               "kernel": ["linear"]},
              {"C": [1,10,100,1000], 
               "kernel": ["rbf"],
               "gamma": [0.62,0.64,0.66,0.68,0.72,0.74,0.70,0.76]}]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
