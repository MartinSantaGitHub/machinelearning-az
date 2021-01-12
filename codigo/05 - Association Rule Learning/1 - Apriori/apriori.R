# Apriori

# Preprocesado de Datos
library(arules)
library(arulesViz)
dataset = read.csv("Market_Basket_Optimisation.csv",header = FALSE)
dataset = read.transactions("Market_Basket_Optimisation.csv",
                            sep = ",",
                            rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset,topN=10)

# Entrenar algoritmo Apriori con el dataset
rules = apriori(data = dataset, 
                parameter = list(support = round(4*7/7500,3), confidence = 0.8/4))

# Visualización de los resultados
inspect(sort(rules, by = 'lift',decreasing = TRUE)[1:10])

# Visualizations
plot(rules, method = "graph", engine = "htmlwidget")
