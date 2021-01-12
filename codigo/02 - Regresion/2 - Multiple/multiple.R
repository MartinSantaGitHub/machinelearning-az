# Regresión Lineal Múltiple

# Importar el data set
dataset = read.csv("2 - Regresion/2 - Multiple/50_Startups.csv")

# Codificar las variables categóricas
dataset$State = factor(dataset$State,
                       levels = c("New York","California","Florida"),
                       labels = c(1,2,3))

# Dividir los datos en conjunto de entrenamiento y en conjunto de testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit,SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

# Ajustar el modelo de Regresión Lineal Múltiple con el Conjunto de Entrenamiento
regression = lm(formula = Profit ~ .,data = training_set)

# Predecir los resultados con el conjunto de testing
y_pred = predict(regression, newdata = testing_set)

# Construir un modelo óptimo con la Eliminación hacia atrás
regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = dataset)
summary(regression)

regression = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = dataset)
summary(regression)

# Eliminación hacia atrás automática utilizando solamente p-valores
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
backwardElimination(dataset, SL)
