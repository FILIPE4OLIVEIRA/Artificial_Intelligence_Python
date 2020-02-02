# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:07:58 2020
@author: engoliveira

"""

# Artificial Intelligence with Keras

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as bk
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Flatten

## TRATANDO AS IMAGENS DO BANCO DE DADOS
max_ind = 39

X = []
Y = []

for numb in range(10):
    rep_numb = str(numb)  # número representado
    for i in range(1,max_ind+1):
        if (i<10):
            ind = '0' + str(i) # indivíduo do teste
        else:
            ind = str(i)
            
        for j in range(1,4):
            test = str(j) # ensaio testado  
            
            # lendo arquivo de imagem
            img = eval("Image.open('" + rep_numb + "_" + ind + "_" + test + ".png').convert('L')")
            
            # convertendo imagem em array
            pixels = np.array(img)
            tonalidade = 240
            
            # Retirando execessos horizontais e veticais
            nz = (pixels > tonalidade).sum(1)
            q = pixels[nz != pixels.shape[1], :]
            
            nz = (q > tonalidade).sum(0)
            q = q[:, nz != q.shape[0]]
            
            pixels = q
            
            # Transformando Array em Imagem
            img = Image.fromarray(pixels)
            
            # Redimensionando Imagem
            newsize = (32,32)
            img = img.resize(newsize)
            
            # Convertendo Imagem para Array
            pixels = np.array(img)
            
            pixels=(pixels < 230).astype(np.int)
            
            pixels = np.reshape(pixels,(32,32))
            
            # Adicionando a Imagem Tratada as listas            
            X.append(pixels)
            Y.append(numb)           
bk.set_image_data_format('channels_first')

## VIZUALIZANDO UMA IMAGEM ALEATÓRIA 
img = randrange(1170)
plt.imshow(X[img], cmap='gray')
print("O número desse dígito é: {}".format(Y[img]))
print("A dimensão da matriz é: {}".format(pixels.shape))
plt.show()

## SEPARANDO EM CONJUNTO DE TREINO E TESTE
(x_train, x_test, y_train, y_test) = train_test_split(np.array(X),np.array(Y),test_size = 0.30, random_state = 42)

## REDIMENSIONANDO O ARRAY PARA O PADRÃO 2D DO KERAS
x_train = x_train.reshape(x_train.shape[0], 1, 32, 32).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 1, 32, 32).astype('float32')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

## IA NO KERAS
# Definindo a arquitetura da Rede Neural usando Keras
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 32, 32), activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2))), model.add(Flatten()), model.add(Dropout(0.3))
model.add(Dense(512, activation="tanh"))    # camada com 512 neurônios
model.add(Dense(256, activation="tanh"))    # camada com 256 neurônios
model.add(Dense(128, activation="tanh"))    # camada com 128 neurônios
model.add(Dense(64,  activation="tanh"))    # camada com 064 neurônios
model.add(Dense(32,  activation="tanh"))    # camada com 032 neurônios
model.add(Dense(10,  activation="softmax")) # camada de saída com 10 neurônios

# Treinando o Modelo usando SGD (Stochastic Gradient Descent)
print("[INFO] treinando a rede neural...")
model.compile(optimizer=SGD(0.01), loss="categorical_crossentropy",metrics=["accuracy"])
H = model.fit(x_train,y_train, batch_size=600, epochs=300, verbose=0,validation_data=(x_test,y_test))

# Avaliando a Rede Neural
print("[INFO] avaliando a rede neural...")
predictions = model.predict(x_test, batch_size=351)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))
scores = model.evaluate(x_test, y_test, verbose=0)
print("\nAccuracy: %.2f%%" % (scores[1]*100))