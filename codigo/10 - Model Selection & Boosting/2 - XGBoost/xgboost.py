# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:11:02 2021

@author: msantamaria
"""

# XGBoost

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values 
y = dataset.iloc[:,13].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], # The column numbers to be transformed (here is [0] but can be [0, 1, 3])    
    remainder='passthrough' # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)
X = X[:,1:]

# Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state = 0)

# Ajustar el modelo XGboost al Conjunto de Entrenamiento
from xgboost import XGBClassifier
classifier = XGBClassifier(objective = "binary:logistic", 
                           use_label_encoder = False, 
                           eval_metric = "logloss")
classifier.fit(X_train, y_train)

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
