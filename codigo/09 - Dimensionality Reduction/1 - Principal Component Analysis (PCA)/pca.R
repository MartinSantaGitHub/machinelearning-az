# ACP

# Regresión Logística

# Importar el data set
dataset = read.csv("Wine.csv")

# Dividir los datos en conjunto de entrenamiento y en conjunto de testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment,SplitRatio = 0.80)
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

# Escalado de valores
training_set[,-14] = scale(training_set[,-14])
testing_set[,-14] = scale(testing_set[,-14])

# Proyección de las componentes principales
library(caret)
library(e1071)
pca = preProcess(x = training_set[,-14], 
                 method = "pca", 
                 pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[,c(2,3,1)] 
testing_set = predict(pca, testing_set)
testing_set = testing_set[,c(2,3,1)] 

# Ajustar el modelo de SVM con el conjunto de entrenamiento
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = "C-classification",
                 kernel = "linear")

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata=testing_set[,-3])

# Crear la matriz de confusión
cm = table(testing_set[,3],y_pred)

library(ElemStatLearn)
# Visualización del conjunto de entrenamiento
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Entrenamiento)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue',ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 3, 'red3', ifelse(set[, 3] == 1, 'green4', 'blue3')))

# Visualización del conjunto de testing
set = testing_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Conjunto de Testing)',
     xlab = 'CP1', ylab = 'CP2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid==2, 'deepskyblue',ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, bg = ifelse(set[, 3] == 3, 'red3', ifelse(set[, 3] == 1, 'green4', 'blue3')))
