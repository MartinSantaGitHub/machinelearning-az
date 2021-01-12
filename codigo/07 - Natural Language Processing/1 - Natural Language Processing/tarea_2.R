# Natural Language Processing

# Importar el data set
dataset_original = read.delim("Restaurant_Reviews.tsv", 
                              quote = "",
                              stringsAsFactors = FALSE)
# Limpieza de textos
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus,FUN = content_transformer(FUN = tolower))
# Consultar el primer elemento del corpus
# as.character(corpus[[1]])
corpus = tm_map(corpus,FUN = removeNumbers)
corpus = tm_map(corpus,FUN = removePunctuation)
corpus = tm_map(corpus,FUN = removeWords, stopwords(kind = "en"))
corpus = tm_map(corpus,FUN = stemDocument)
corpus = tm_map(corpus,FUN = stripWhitespace)

# Crear el modelo Bag of Words
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,0.999)

dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Codificar la variable de clasificación como factor
dataset$Liked = factor(dataset$Liked,
                       levels = c(0,1))

# Dividir los datos en conjunto de entrenamiento y en conjunto de testing
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked,SplitRatio = 0.80)
training_set = subset(dataset,split==TRUE)
testing_set = subset(dataset,split==FALSE)

# Random Forest
library(randomForest)
classifier = randomForest(x = training_set[,-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier,newdata=testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692],y_pred)

# Métricas
TP = cm[2,2]; TN = cm[1,1]; FP = cm[1,2]; FN = cm[2,1] 
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1.score = 2*precision*recall/(precision+recall)

sprintf("Accuracy: %.2f",accuracy)
sprintf("Precision: %.2f",precision)
sprintf("Recall: %.2f",recall)
sprintf("F1 Score: %.2f",f1.score)

# Salida
#Accuracy: 0.80
#Precision: 0.81
#Recall: 0.77
#F1 Score: 0.79

# Regresión Logística
classifier = glm(formula = Liked ~ .,
                 data = training_set,
                 family = binomial)

# Predicción de los resultados con el conjunto de testing
prob_pred = predict(classifier,
                    type="response",
                    newdata=testing_set[,-692])

y_pred = ifelse(prob_pred>0.5,1,0)

# Crear la matriz de confusión
cm = table(testing_set[,692],y_pred)

# Métricas
TP = cm[2,2]; TN = cm[1,1]; FP = cm[1,2]; FN = cm[2,1] 
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1.score = 2*precision*recall/(precision+recall)

sprintf("Accuracy: %.2f",accuracy)
sprintf("Precision: %.2f",precision)
sprintf("Recall: %.2f",recall)
sprintf("F1 Score: %.2f",f1.score)

# Salida
#Accuracy: 0.55
#Precision: 0.54
#Recall: 0.61
#F1 Score: 0.58

# K-NN.
library(class)
y_pred = knn(train = training_set[,-692],
             test = testing_set[,-692],
             cl = training_set[,692],
             k = 5)

# Crear la matriz de confusión
cm = table(testing_set[,692],y_pred)

# Métricas
TP = cm[2,2]; TN = cm[1,1]; FP = cm[1,2]; FN = cm[2,1] 
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1.score = 2*precision*recall/(precision+recall)

sprintf("Accuracy: %.2f",accuracy)
sprintf("Precision: %.2f",precision)
sprintf("Recall: %.2f",recall)
sprintf("F1 Score: %.2f",f1.score)

# Salida
#Accuracy: 0.64
#Precision: 0.68
#Recall: 0.54
#F1 Score: 0.60

# SVM
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = "C-classification",
                 kernel = "linear")

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata=testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692],y_pred)

# Métricas
TP = cm[2,2]; TN = cm[1,1]; FP = cm[1,2]; FN = cm[2,1] 
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1.score = 2*precision*recall/(precision+recall)

sprintf("Accuracy: %.2f",accuracy)
sprintf("Precision: %.2f",precision)
sprintf("Recall: %.2f",recall)
sprintf("F1 Score: %.2f",f1.score)

# Salida
#Accuracy: 0.80
#Precision: 0.79
#Recall: 0.81
#F1 Score: 0.80

# Kernel SVM
library(e1071)
classifier = svm(formula = Liked ~ .,
                 data = training_set,
                 type = "C-classification",
                 kernel = "radial")

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier,newdata = testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692],y_pred)

# Métricas
TP = cm[2,2]; TN = cm[1,1]; FP = cm[1,2]; FN = cm[2,1] 
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1.score = 2*precision*recall/(precision+recall)

sprintf("Accuracy: %.2f",accuracy)
sprintf("Precision: %.2f",precision)
sprintf("Recall: %.2f",recall)
sprintf("F1 Score: %.2f",f1.score)

# Salida
#Accuracy: 0.53
#Precision: 1.00
#Recall: 0.06
#F1 Score: 0.11

# Naive Bayes
library(e1071)
classifier = naiveBayes(x = training_set[,-692],
                        y = training_set$Liked)

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier,newdata=testing_set[,-692])

# Crear la matriz de confusión
cm = table(testing_set[,692],y_pred)

# Métricas
TP = cm[2,2]; TN = cm[1,1]; FP = cm[1,2]; FN = cm[2,1] 
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1.score = 2*precision*recall/(precision+recall)

sprintf("Accuracy: %.2f",accuracy)
sprintf("Precision: %.2f",precision)
sprintf("Recall: %.2f",recall)
sprintf("F1 Score: %.2f",f1.score)

# Salida
#Accuracy: 0.50
#Precision: 0.50
#Recall: 0.96
#F1 Score: 0.66

# Decision Tree
library(rpart)
classifier = rpart(formula = Liked ~ .,
                   data = training_set)

# Predicción de los resultados con el conjunto de testing
y_pred = predict(classifier, newdata=testing_set[,-692], type = "class")

# Crear la matriz de confusión
cm = table(testing_set[,692],y_pred)

# Métricas
TP = cm[2,2]; TN = cm[1,1]; FP = cm[1,2]; FN = cm[2,1] 
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
f1.score = 2*precision*recall/(precision+recall)

sprintf("Accuracy: %.2f",accuracy)
sprintf("Precision: %.2f",precision)
sprintf("Recall: %.2f",recall)
sprintf("F1 Score: %.2f",f1.score)

# Salida
#Accuracy: 0.71
#Precision: 0.79
#Recall: 0.57
#F1 Score: 0.66

#Se observan muy buenos resultados empleando el algoritmo de SVM
