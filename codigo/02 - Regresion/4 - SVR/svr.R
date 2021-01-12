# SVR

# Importar el data set
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

# Ajustar SVR con el Conjunto de Datos
library(e1071)
regression = svm(formula = Salary ~ ., 
                 data = dataset, 
                 type = "eps-regression",
                 kernel = "radial")

# Predicción de nuevos resultados con Regresión SVR
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

# Visualización del modelo de SVR
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(regression, newdata = data.frame(Level = dataset$Level))), 
            color = "blue") + 
  ggtitle("Predicción (SVR)") + 
  xlab("Nivel del empleado") + 
  ylab("Sueldo (en dolares)")
