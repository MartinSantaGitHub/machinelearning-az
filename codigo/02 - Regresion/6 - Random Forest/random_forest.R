# Bosques Aleatorios

# Importar el data set
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

# Dividir los datos en conjunto de entrenamiento y en conjunto de testing

# Escalado de valores

# Ajustar Modelo de Bosques Aleatorios con el Conjunto de Datos
library(randomForest)
set.seed(1234)
# regression = randomForest(formula = Salary ~ Level, 
#                           data = dataset, 
#                           ntree = 10)
regression = randomForest(x = dataset[1],
                          y = dataset$Salary,
                          ntree = 1000)

# Predicción de nuevos resultados con Bosques Aleatorios
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

# Visualización del modelo de Bosques Aleatorios
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.01)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, newdata = data.frame(Level = x_grid))), 
            color = "blue") + 
  ggtitle("Predicción del sueldo en función del nivel del empleado") + 
  xlab("Nivel del empleado") + 
  ylab("Sueldo (en dolares)")
