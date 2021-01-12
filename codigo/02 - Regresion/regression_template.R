# Regresión

# Importar el data set
dataset = read.csv("2 - Regresion/3 - Polinomica/Position_Salaries.csv")
dataset = dataset[,2:3]

# Dividir los datos en conjunto de entrenamiento y en conjunto de testing

# Escalado de valores

# Ajustar Modelo de Regresión con el Conjunto de Datos
# Crear nuestra variable de regresión aqui

# Predicción de nuevos resultados con Regresión
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

# Visualización del modelo de regresión
library(ggplot2)
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = x_grid, y = predict(regression, newdata = data.frame(Level = x_grid))), 
            color = "blue") + 
  ggtitle("Predicción del sueldo en función del nivel del empleado") + 
  xlab("Nivel del empleado") + 
  ylab("Sueldo (en dolares)")
