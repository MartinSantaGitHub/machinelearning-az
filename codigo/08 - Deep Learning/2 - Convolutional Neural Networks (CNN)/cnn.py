# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:53:53 2021

@author: msantamaria
"""

# Redes Neuronales Convolucionales

# Parte 1 - Construir el modelo de CNN

# Importar las librerías y paquetes
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Conv2D(filters = 32, 
                      kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), 
                      activation = "relu"))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 32, 
                      kernel_size = (3, 3),                       
                      activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

# Una tercera capa de convolución y max pooling
classifier.add(Conv2D(filters = 32, 
                      kernel_size = (3, 3),                       
                      activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full Connection
classifier.add(Dense(units = 128,
                     activation = "relu"))
classifier.add(Dense(units = 1,
                     activation = "sigmoid"))

# Compilar la CNN
classifier.compile(optimizer="adam", 
                   loss="binary_crossentropy", 
                   metrics=["accuracy"])

# Parte 2 - Ajustar la CNN a las imágenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='binary')

classifier.fit(training_dataset,
               steps_per_epoch=int(8000/32),
               epochs=35,
               validation_data=testing_dataset,
               validation_steps=int(2000/32))

# For a single prediction
import numpy as np
from keras.preprocessing import image
 
test_image = image.load_img("dataset/test_image/cat.4008.jpg",
                            target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict_on_batch(test_image)
#training_dataset.class_indices

print(result[0][0])

if result[0][0] > 0.5: 
    print("perro")
else:
    print("gato")
