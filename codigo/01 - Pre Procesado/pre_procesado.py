# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 21:02:22 2020

@author: msantamaria
"""

# Plantilla de Pre Procesado

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:,:-1].values # iloc -> index localization
y = dataset.iloc[:,-1].values

# Tratamiento de los NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, 
                        strategy = "mean")
imputer = imputer.fit(X[:,1:3]) 
X[:,1:3] = imputer.transform(X[:,1:3])
#X[:,1:3] = imputer.fit_transform(X[:,1:3])

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], # The column numbers to be transformed (here is [0] but can be [0, 1, 3])    
    remainder='passthrough' # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype=np.float)
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)
 
 # Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
