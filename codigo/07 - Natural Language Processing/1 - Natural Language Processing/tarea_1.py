# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 13:32:35 2021

@author: msantamaria
"""

# Natural Language Processing

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

# Limpieza de texto
import re
#import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):    
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

 # Dividir el dataset en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state = 0)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)

# Métricas
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {2*precision*recall/(precision+recall):.2f}")

# Salida
#Accuracy: 0.73
#Precision: 0.68
#Recall: 0.88
#F1 Score: 0.77

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver="liblinear",multi_class="ovr",n_jobs=1,random_state=0)
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión 
cm = confusion_matrix(y_test,y_pred)

# Métricas
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {2*precision*recall/(precision+recall):.2f}")

# Salida
#Accuracy: 0.71
#Precision: 0.76
#Recall: 0.64
#F1 Score: 0.69

# K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,
                                  metric='minkowski',
                                  p=2)
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
cm = confusion_matrix(y_test,y_pred)

# Métricas
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {2*precision*recall/(precision+recall):.2f}")

# Salida
#Accuracy: 0.61
#Precision: 0.68
#Recall: 0.47
#F1 Score: 0.55

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state=0)
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión 
cm = confusion_matrix(y_test,y_pred)

# Métricas
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {2*precision*recall/(precision+recall):.2f}")

# Salida
#Accuracy: 0.72
#Precision: 0.75
#Recall: 0.68
#F1 Score: 0.71

# Kernel SVM
from sklearn.svm import SVC
classifier = SVC(kernel = "rbf", gamma = "scale", random_state=0)
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
cm = confusion_matrix(y_test,y_pred)

# Métricas
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {2*precision*recall/(precision+recall):.2f}")

# Salida
#Accuracy: 0.73
#Precision: 0.89
#Recall: 0.55
#F1 Score: 0.68

# Decision Tree 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
cm = confusion_matrix(y_test,y_pred)

# Métricas
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {2*precision*recall/(precision+recall):.2f}")

# Salida
#Accuracy: 0.71
#Precision: 0.75
#Recall: 0.66
#F1 Score: 0.70

# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)

# Elaborar una matriz de confusión
cm = confusion_matrix(y_test,y_pred)

# Métricas
TP, TN, FP, FN = cm[1,1], cm[0,0], cm[0,1], cm[1,0]
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print(f"Accuracy: {(TP+TN)/(TP+TN+FP+FN):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {2*precision*recall/(precision+recall):.2f}")

# Salida
#Accuracy: 0.72
#Precision: 0.85
#Recall: 0.55
#F1 Score: 0.67

#Se observan resultados interesantes empleando el modelo Kernel SVM
