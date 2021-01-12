# Regresión Lineal Simple

# Importar el data set
dataset = read.csv("2 - Regresion/1 - Simple/Salary_Data.csv")

# Dividir los datos en conjunto de entrenamiento y en conjunto de testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3)
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

# Ajustar el modelo de regresión lineal simple con el conjunto de entrenamiento
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# Predecir resultados con el conjunto de test
y_pred = predict(regressor,newdata = testing_set)

# Visualización de los resultados en el conjunto de entrenamiento
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience,
                 y = training_set$Salary),
             colour='red') + 
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor,newdata = training_set)),
            colour='blue') +
  ggtitle("Sueldo vs Años de Experiencia (Conunto de Entrenamiento)") + 
  xlab("Años de experiencia") + 
  ylab("Sueldo (en dolares)")

# Visualización de los resultados en el conjunto de testing
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience,
                 y = testing_set$Salary),
             colour='red') + 
  geom_line(aes(x = training_set$YearsExperience, 
                y = predict(regressor,newdata = training_set)),
            colour='blue') +
  ggtitle("Sueldo vs Años de Experiencia (Conunto de Testing)") + 
  xlab("Años de experiencia") + 
  ylab("Sueldo (en dolares)")

