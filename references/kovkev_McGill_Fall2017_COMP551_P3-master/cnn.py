'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

from keras import backend as K

batch_size = 128
num_classes = 82
# num_classes = 10
epochs = 120

# input image dimensions
img_rows, img_cols = 64, 64
# img_rows, img_cols = 28, 28


def get_dataset():

  dataset = [
    (
      [5],
      [6],
    )
  ]
  # validate_data(dataset)
  import numpy   as np
  import scipy.misc # to visualize only
  y = np.loadtxt("train_y.csv", delimiter=",")

  y = y.reshape(-1)
  y = y.astype(int)[:10000]

  n_values = np.max(y) + 1
  one_hot_ks = np.eye(n_values)[y]

  print("loading train_x.csv")
  import pandas as pd
  xxs = pd.read_csv("train_x.csv", delimiter=",", header=None)
  xxs = np.array(xxs)[:10000]
  print("loaded train_x.csv")
  xxs = xxs.reshape(-1, 64, 64)
  xxs = xxs.astype('f')

  out = (xxs, y)

  print("returning")

  return out

  y_hot_k[y] = 1
  print("A")
  # scipy.misc.imshow(x[0]) # to visualize only

  return dataset


import numpy as np

def get_train_and_validation():
  # return mnist.load_data()

  dataset_x, dataset_y = get_dataset()
  validation_set_proportion = 0.2
  validation_set_size = int(int(len(dataset_y)) * validation_set_proportion)
  # np.random.shuffle(dataset)
  training_set_x, validation_set_x = dataset_x[:-validation_set_size], dataset_x[-validation_set_size:]
  training_set_y, validation_set_y = dataset_y[:-validation_set_size], dataset_y[-validation_set_size:]
  return (training_set_x, training_set_y), (validation_set_x, validation_set_y)
  # return get_dataset

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = get_train_and_validation()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(256, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(
                  lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
              # keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


score = model.evaluate(x_test, y_test, verbose=0)
# model.save()

import pandas as pd
test_set_x = pd.read_csv("test_x.csv", delimiter=",", header=None)
test_set_x = np.array(test_set_x)
print("loaded test_x.csv")
test_set_x = test_set_x.reshape(-1, 64, 64, 1)
test_set_x = test_set_x.astype('float32')
test_set_x /= 255

prediction = model.predict(test_set_x)

digits = np.array([ np.argmax(a) for a in prediction ]).astype(int)
np.savetxt("predictions.csv", digits, delimiter=",", fmt="%d")

print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy")
# plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss")
# plt.show()
