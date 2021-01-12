# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 23:42:28 2021

@author: msantamaria
"""

# Apriori

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set 
dataset = pd.read_csv("Market_Basket_Optimisation.csv",header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Entrenar el algoritmo de Apriori
from apyori import apriori
rules = apriori(transactions, 
                min_support = 0.003, 
                min_confidence = 0.2, 
                min_lift = 3, 
                min_length = 2)

# Visualización de los resultados
results = list(rules)
results[0]

the_rules = []
for result in results:
    the_rules.append({'rule': ','.join(result.items),
                      'support':result.support,
                      'confidence':result.ordered_statistics[0].confidence,
                      'lift':result.ordered_statistics[0].lift})
df = pd.DataFrame(the_rules, columns = ['rule', 'support', 'confidence', 'lift'])
