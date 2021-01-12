# Regresión Polinómica

# Importar el data set
dataset = read.csv("2 - Regresion/3 - Polinomica/Position_Salaries.csv")
dataset = dataset[,2:3]

# Ajustar modelo de regresión lineal con el conjunto de datos
lin_reg = lm(formula = Salary ~ ., data = dataset)

# Ajustar Modelo de Regresión Polinómica con el Conjunto de Datos
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)

# Visualización del modelo lineal
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), color = "blue") + 
  ggtitle("Predicción lineal del sueldo en función del nivel del empleado") + 
  xlab("Nivel del empleado") + 
  ylab("Sueldo (en dolares)")

# Visualización del modelo polinómico
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color = "red") +
  geom_line(aes(x = x_grid, y = predict(poly_reg, newdata = data.frame(Level = x_grid,
                                                                       Level2 = x_grid^2,
                                                                       Level3 = x_grid^3,
                                                                       Level4 = x_grid^4))), 
            color = "blue") + 
  ggtitle("Predicción lineal del sueldo en función del nivel del empleado") + 
  xlab("Nivel del empleado") + 
  ylab("Sueldo (en dolares)")

# Predicción de nuevos resultados con Regresión Lineal
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))

# Predicción de nuevos resultados con Regresión Polinómica
y_pred_poly = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                     Level2 = 6.5^2,
                                                     Level3 = 6.5^3,
                                                     Level4 = 6.5^4))
