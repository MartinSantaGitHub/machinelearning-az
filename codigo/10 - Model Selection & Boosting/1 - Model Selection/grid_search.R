# Grid Search

# Importar el data set
dataset = read.csv("Social_Network_Ads.csv")
dataset = dataset[,3:5]

dataset$Purchased = factor(dataset$Purchased)

# Dividir los datos en conjunto de entrenamiento y en conjunto de testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.75)
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

# Escalado de valores
training_set[,1:2] = scale(training_set[,1:2])
testing_set[,c(1,2)] = scale(testing_set[,c(1,2)])

# Ajustar el clasificador con el conjunto de entrenamiento.
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = "C-classification",
                 kernel = "radial")

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier,newdata = testing_set[,-3])

# Crear la matriz de confusión
cm = table(testing_set[,3],y_pred)

# Aplicar algoritmo k-fold cross validation
library(caret)
folds = createFolds(training_set$Purchased, k = 10)
cv = lapply(folds, function(x){
  training_fold = training_set[-x,]
  test_fold = training_set[x,]
  classifier = svm(formula = Purchased ~ .,
                   data = training_fold,
                   type = "C-classification",
                   kernel = "radial")
  y_pred = predict(classifier,newdata = test_fold[,-3])
  cm = table(test_fold[,3],y_pred)
  accuracy = (cm[1,1]+cm[2,2])/(cm[1,1]+cm[2,2]+cm[2,1]+cm[1,2])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy_sd = sd(as.numeric(cv))

# Aplicar Grid Search para encontrar los parámetros óptimos
library(caret)
classifier = train(form = Purchased ~ .,
                   data = training_set,
                   method = "svmRadial")
classifier$bestTune
