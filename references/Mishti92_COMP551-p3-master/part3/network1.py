
# coding: utf-8

# Convolutional Neural Networks
# 
# • Model with a large learning capacity
# 
# • Prior knowledge to compensate all data we do not have

# In[1]:

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')


# In[2]:

import keras
import numpy as np
from matplotlib import pyplot as plt
# import lasagne


# In[42]:

file_path = "../data/"
training_data = np.load(file_path+"tinyX.npy") #(26344, 3, 64, 64)
training_label = np.load(file_path+"tinyY.npy") #(26344,)
test_data = np.load(file_path+"tinyX_test.npy") #(6600, 3, 64, 64)


# In[43]:

training_label.shape


# In[44]:

training_data = training_data.transpose([0, 3, 2, 1])
test_data = test_data.transpose([0, 3, 2, 1])


# In[101]:

# plt.imshow(training_data[0])
# plt.show()


# In[46]:

# # to visualize only
# import scipy.misc
# scipy.misc.imshow(training_data[0]) # put RGB channels last


# In[100]:

nbclasses = np.unique(training_label).shape[0]
# print "Number of classes: ", nbclasses 
# Number of classes:  40
plt.hist(training_label, bins=40, normed=True)
plt.title("Normalized Distribution of Data")
# plt.show()


# In[48]:

image_shape = training_data[0].reshape([-1]).shape[0] #12288
np.random.seed(123)  # for reproducibility


# In[49]:

#a linear stack of neural network layers, and it's perfect for the type of feed-forward CNN
from keras.models import Sequential
#import the "core" layers from Keras. These are the layers that are used in almost any neural network:
from keras.layers import Dense, Dropout, Activation, Flatten
#import the CNN layers from Keras. These are the convolutional layers that will help us efficiently train on image data:
from keras.layers import Convolution2D, MaxPooling2D
#import some utilities. This will help us transform our data later
from keras.utils import np_utils


# In[97]:

batch_size = 128
nb_classes = 10
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 64, 64
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# Noise sigma
sigma = 0.2

# change this
epochs = 1

# training_data.shape (26344, 64, 64, 3)


# In[96]:

# training_label.shape (26344,)


# In[94]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_data,training_label, test_size=0.30, random_state=30)
# X_train.shape (18440, 64, 64, 3)


# In[53]:

#convert our data type to float32 and normalize our data values to the range [0, 1].
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_predict = test_data.astype('float32')
X_train /= 255
X_test /= 255


# In[93]:

# print X_train.shape
# print y_train.shape
# plt.hist(y_test[:1000], bins=40)
# plt.title("Prediction Distribution")
# plt.show()

# (18440, 64, 64, 3)
# (18440,)


# **Preprocess class labels for Keras**
# 
# We should have 40 different classes, one for each digit, but it looks like we only have a 1-dimensional array.

# In[55]:

# Convert 1-dimensional class arrays to 40-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 40)
Y_test = np_utils.to_categorical(y_test, 40)


# In[92]:

# print X_train.shape
# print X_predict.shape

# (18440, 64, 64, 3)
# (6600, 64, 64, 3)


# In[57]:

# X_train = X_train[:3000]
# Y_train = Y_train[:3000]
# X_test = X_train[:1000]
# Y_test = Y_train[:1000]


# In[58]:

# Y_train


# In[59]:

from keras import backend as K

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    X_predict = X_predict.reshape(X_predict.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_predict = X_predict.reshape(X_predict.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)


# In[60]:

# from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

# model = Sequential()
# model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
# model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(40, activation='softmax'))


# In[61]:

# model.output_shape


# In[62]:

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])


# In[63]:

# # this will do preprocessing and realtime data augmentation
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=False,  # randomly flip images
#     vertical_flip=False)  # randomly flip images

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# img = load_img('../data/test1/10.jpg')  # this is a PIL image
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,
#                           save_to_dir='../data/preview', save_prefix='cat', save_format='jpeg'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely


# In[90]:

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# the model so far outputs 3D feature maps (height, width, features)

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(40, activation='softmax'))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))




# In[91]:

# print model.output_shape
# print X_train.shape
# print X_test.shape
# print X_predict.shape
# print Y_train.shape
# # Y_train

# (None, 40)
# (18440, 3, 64, 64)
# (7904, 3, 64, 64)
# (6600, 3, 64, 64)
# (18440, 40)


# In[84]:

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[85]:

datagen.fit(X_train)
# datagen.fit(X_test)


# In[86]:

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(X_train, Y_train,
                                 batch_size=batch_size),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))
model.save_weights('../weights/cnnTry.h5')  


# In[204]:

classes = model.predict(X_predict, batch_size=32)


# In[205]:

classes = classes.argmax(axis = 1)


# In[206]:

# Saves the prediction file in prediction folder.
import csv
test_predict_CNN_filename = "../predictions/CNN_predict"
with open(test_predict_CNN_filename + time.strftime("%d_%m_%Y_%H_%M_%S") + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'class'])

        for i in range(classes.shape[0]):

            writer.writerow([i, classes[i]])


# In[87]:

# plt.hist(classes, bins=40)
# plt.title("Prediction Distribution")
# plt.show()


# In[ ]:




# In[88]:

# model.summary()


# In[39]:




# In[36]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



