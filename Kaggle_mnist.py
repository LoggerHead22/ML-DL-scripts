# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 00:08:33 2020

@author: Logge
"""

import pandas as pd
import tensorflow 
import csv
from tensorflow.keras.datasets import boston_housing
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers


data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_data = np.array(data)
test_data = np.array(test)

labels = train_data[:, 0].astype('float')
train_data = train_data[:, 1:].astype('float') / 255.0
test_data = test_data.astype('float') / 255.0

train_data = train_data.reshape((42000,28,28,1))
test_data = test_data.reshape((28000,28,28,1))

from keras.utils import to_categorical

labels = to_categorical(labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_data, labels, epochs=6, batch_size=64)



#%%
y_pred = model.predict(test_data)

#%%
y_preds = np.array(list(map(lambda i: np.argmax(i) , y_pred)))

with open('submission.csv', 'w' ,newline='') as f:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    i = 1
    for elem in y_preds:  
        writer.writerow({'ImageId': i, 'Label': elem})
        i += 1




