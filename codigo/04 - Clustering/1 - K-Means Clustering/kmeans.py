# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:06:56 2020

@author: msantamaria
"""

# K-Means

# Importar las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargamos los datos con pandas
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values

# Metodo del codo (elbow) para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("Metodo del codo")
plt.xlabel("Numero de Clusters")
plt.ylabel("WCSS(k)")
plt.show()

# Aplicar el metodo de k-means para segmentar el data set
kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Clusters information
#kmeans.cluster_centers_
#kmeans.labels_
#
#Cluster_0 = X[y_kmeans == 0]
#Cluster_1 = X[y_kmeans == 1]
#Cluster_2 = X[y_kmeans == 2]
#Cluster_3 = X[y_kmeans == 3]
#Cluster_4 = X[y_kmeans == 4]
#
#len(Cluster_0)
#len(Cluster_1)
#len(Cluster_2)
#len(Cluster_3)
#len(Cluster_4)
#
#Cluster_0[0].mean()
#Cluster_1[0].mean()
#Cluster_2[0].mean()
#Cluster_3[0].mean()
#Cluster_4[0].mean()
#Cluster_0[0].std()
#Cluster_1[0].std()
#Cluster_2[0].std()
#Cluster_3[0].std()
#Cluster_4[0].std()
#
#len(Cluster_0) / 200
#len(Cluster_1) / 200
#len(Cluster_2) / 200
#len(Cluster_3) / 200
#len(Cluster_4) / 200

# Visualizacion de los clusters 
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100,c="red",label="Cautos")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100,c="blue",label="Estandar")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100,c="green",label="Objetivo")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s=100,c="cyan",label="Descuidados")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s=100,c="magenta",label="Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="yellow",label="Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuacion de Gastos (1-100)")
plt.legend()
plt.show()
