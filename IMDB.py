import tensorflow 
from tensorflow.keras.datasets import imdb
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers


def vectorize_sequences(sequences, dimension = 10000):
  results = np.zeros((len(sequences) , dimension))
  for i,sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results 

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(len(train_data))


X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = models.Sequential()
model.add(layers.Dense(16,activation='relu', input_shape =(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

X_val = X_train[:10000]
partial_X_train = X_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = model.fit(partial_X_train,partial_y_train,epochs=20,batch_size=512, validation_data=(X_val, y_val))


#%%
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
model.fit(X_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(X_test, y_test , verbose = 0)

print(results)











