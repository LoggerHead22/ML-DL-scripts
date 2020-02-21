# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:49:16 2020

@author: Logge
"""

import tensorflow 
from tensorflow.keras.datasets import reuters
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers

def vectorize_sequences(sequences, dimension = 10000):
  results = np.zeros((len(sequences) , dimension))
  for i,sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results 

def to_one_hot(labels, dimension=46):
  results = np.zeros((len(labels), dimension))
  for i, label in enumerate(labels):
    results[i, label] = 1.
  return results


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

y_train = to_one_hot(train_labels) # from keras.utils.np_utils import to_categorical
y_test = to_one_hot(test_labels)  # one_hot_train_labels = to_categorical(train_labels)

model = models.Sequential()
model.add(layers.Dense(64,activation='relu', input_shape =(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46, activation='softmax')) # softmax - вероятность каждого класса 

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy']) #sparse_categorial_crossentropy (если не кодируем метки)

X_val = X_train[:1000]
partial_X_train = X_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]


history = model.fit(partial_X_train,partial_y_train,epochs=20,batch_size=512, validation_data=(X_val, y_val))

import matplotlib.pyplot as plt

history_dict = history.history
print(history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo',label ='Training acc')
plt.plot(epochs, val_acc_values, 'b',label ='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuaracy')
plt.legend()

plt.show()

#%%
#model.fit(partial_X_train,partial_y_train,epochs=9,batch_size=512, validation_data=(X_val, y_val))


model.fit(X_train, y_train, epochs=9, batch_size=512)
results = model.evaluate(X_test, y_test , verbose = 0)

print(results)

predictions = model.predict(X_test) # np.argmax(prediction[i])


