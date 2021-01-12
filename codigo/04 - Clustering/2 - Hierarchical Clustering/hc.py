#!C:\Python38 python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:30:43 2021

@author: msantamaria
"""

# Clustering Jerárquico

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar los datos del centro comercial 
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# Se debería escalar la variable X (se utilizan distancias (euclideas) en el modelo)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Utilizar el dendrograma para encontrar el número óptimo de clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclídea")
plt.show()

# Ajustar el clustering jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc = hc.fit_predict(X)

# Visualizacion de los clusters 
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s=100,c="red",label="Cautos")
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s=100,c="blue",label="Estandar")
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s=100,c="green",label="Objetivo")
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s=100,c="cyan",label="Descuidados")
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s=100,c="magenta",label="Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de Gastos (1-100)")
plt.legend()
plt.show()
