
# coding: utf-8


import os
import numpy
import scipy
import scipy.misc
from PIL import Image

import random, string

OUTPUT_DIR = "../data/label/"
trainX = numpy.load('../data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = numpy.load('../data/tinyY.npy') 
testX = numpy.load('../data/tinyX_test.npy') # (6600, 3, 64, 64)


# In[ ]:

from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
import keras
import numpy as np
from matplotlib import pyplot as plt

import csv
import keras
import time
import random
import platform
import os
import shutil
import h5py
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adagrad
from keras import backend as K
from keras import initializations

from sklearn.model_selection import KFold
from scipy import misc
from PIL import Image


# In[ ]:

print "Setting file paths and directories"

file_path = "../data/"
training_data = np.load(file_path+"tinyX.npy").transpose([0, 3, 2, 1])
training_label = np.load(file_path+"tinyY.npy") #(26344,)
test_data = np.load(file_path+"tinyX_test.npy").transpose([0, 3, 2, 1])

test_predict_CNN_filename = "test_predict_CNN_weights_"
saved_model_filename = "saved_model"


root_folder_name = "../data_CNN/"
root_folder_name_for_prediction = '../predictions_CNN/'

weights_path = '../weights/vgg16_weights.h5'
model_weights_path = '../weights/features_bottleneck_model.h5'
model_weights_path_for_prediction = '../weights/features_bottleneck_prediction.h5'

train_folder_name = root_folder_name + 'train/'
val_folder_name = root_folder_name + 'val/'

train_index_filename = "../index/train_index.npy"
validation_index_filename = "../index/validation_index.npy"

train_index_filename_for_prediction = "train_index_for_prediction.npy"
validation_index_filename_for_prediction = "validation_index_for_prediction.npy"
train_data_dir = root_folder_name + 'train'
validation_data_dir = root_folder_name + 'val'

train_data_dir_for_prediction = root_folder_name_for_prediction + 'train_for_prediction/'
validation_data_dir_for_prediction = root_folder_name_for_prediction + 'val_for_prediction/'
test_data_dir_for_prediction = root_folder_name_for_prediction + 'test_for_prediction/'

print "Done............."


num_of_classes = np.unique(training_label).shape[0]
image_size = training_data[0].reshape([-1]).shape[0]

print("num_of_classes: %d" % (num_of_classes))
print("image_shape: %d" % (image_size))

print "Generating training_set_label_matrix"
training_set_label_matrix = np.zeros([training_label.shape[0], num_of_classes])

for i in range(training_label.shape[0]):
    training_set_label_matrix[i][training_label[i]] = 1
    
print(training_set_label_matrix.shape)

print "Done................"


# In[5]:

img_width = 64
img_height = 64
nb_epoch = 50

save_model = True
root_folder_name = "../data_CNN/"
root_folder_name_for_prediction = '../predictions_CNN/'

weights_path = '../weights/vgg16_weights.h5'
top_model_weights_path = '../weights/bottleneck_fc_model.h5'
top_model_weights_path_for_prediction = '../weights/bottleneck_fc_model_for_prediction.h5'

train_folder_name = root_folder_name + 'train/'
val_folder_name = root_folder_name + 'val/'

train_index_filename = "../index/train_index.npy"
validation_index_filename = "../index/validation_index.npy"

train_index_filename_for_prediction = "train_index_for_prediction.npy"
validation_index_filename_for_prediction = "validation_index_for_prediction.npy"
train_data_dir = root_folder_name + 'train'
validation_data_dir = root_folder_name + 'val'

train_data_dir_for_prediction = root_folder_name_for_prediction + 'train_for_prediction/'
validation_data_dir_for_prediction = root_folder_name_for_prediction + 'val_for_prediction/'
test_data_dir_for_prediction = root_folder_name_for_prediction + 'test_for_prediction/'


# In[9]:

print("Parameter setting.")

img_width = 64
img_height = 64
nb_epoch = 1

save_model = True

first_try = True

if first_try == True:
    generate_k_fold = True
    calculate_features = True
    calculate_features_for_prediction = True
else:
    generate_k_fold = False
    calculate_features = False
    calculate_features_for_prediction = False
    
mode = "Predict"

k = 5

print "Done.................."


# In[10]:

print("Generate the training set and test set.")


print(mode + " the model.")
print(str(k) + "-fold cross validation.")


if mode == "Evaluate":
    if generate_k_fold == True:
        k_fold = KFold(n_splits = k, shuffle = True, random_state = np.random.RandomState())
        train_index, validation_index = next(k_fold.split(training_set_data_for_use))
        
        np.save(train_index_filename, train_index)
        np.save(validation_index_filename, validation_index)
    else:
        train_index = np.load(train_index_filename)
        validation_index = np.load(validation_index_filename)
else:
    train_index = np.array([i for i in range(training_set_data_for_use.shape[0])])
    validation_index = np.array([])
    

def create_folder(folder_name, delete = False):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    elif delete == True:
        shutil.rmtree(folder_name)
        os.makedirs(folder_name)

if mode == "Evaluate":
    if generate_k_fold == True:
        create_folder(root_folder_name, True)
        create_folder(train_folder_name, True)
        create_folder(val_folder_name, True)
        print("Create the folders to store the generated images.")
else:
    if generate_k_fold == True:
        create_folder(root_folder_name_for_prediction, True)
        create_folder(train_data_dir_for_prediction, True)
        create_folder(validation_data_dir_for_prediction, True)
        create_folder(test_data_dir_for_prediction, True)
        
        print("Create the folders to store the generated images.")
        
        
print "Done..............."


# In[13]:

print("Writing the images to the corresponding folders.")

print("Record the time.\n")
start_time = time.time()

labels_train = training_set_label_matrix[train_index]
labels_val = np.array([])

if mode == "Evaluate":
    folder_to_write_name = train_folder_name
else:
    folder_to_write_name = train_data_dir_for_prediction
    
    
for i in range(train_index.shape[0]):
    
    index = train_index[i]
    image = training_set_data[index]
    label = training_set_label[index]

    if generate_k_fold == True:
        folder_for_store_the_image = folder_to_write_name + str(label) + "/"
        create_folder(folder_for_store_the_image)

        img = Image.fromarray(image)
        img.save(folder_for_store_the_image + str(i) + ".jpg", 'JPEG')
    
    

if mode == "Evaluate":
    labels_val = training_set_label_matrix[validation_index]
    for i in range(validation_index.shape[0]):

        index = validation_index[i]
        image = training_set_data[index]
        label = training_set_label[index]

        if generate_k_fold == True:
            folder_for_store_the_image = val_folder_name + str(label) + "/"
            create_folder(folder_for_store_the_image)

            img = Image.fromarray(image)
            img.save(folder_for_store_the_image + str(i) + ".jpg", 'JPEG')
            
else:
    for i in range(test_set_data.shape[0]):
    
        image = test_set_data[i]

        if generate_k_fold == True:
            folder_for_store_the_image = test_data_dir_for_prediction + "0/"
            create_folder(folder_for_store_the_image)

            img = Image.fromarray(image)
            img.save(folder_for_store_the_image + str(i) + ".jpg", 'JPEG')

    

    
print("Finish writing the images.")
end_time = time.time()
print("\n--- %s seconds ---" % (end_time - start_time))


# In[14]:

nb_train_samples = train_index.shape[0]
nb_validation_samples = validation_index.shape[0]
nb_test_samples = test_set_data.shape[0]



def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.1, name=name)


def save_bottleneck_features(train_data_dir,
                             validation_data_dir,
                             weights_path, bottleneck_features_train_filename,
                             bottleneck_features_validation_filename,
                             img_width,
                             img_height, save_model = True, is_test = False):

    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    print "here"
#     assert os.path.exists(weights_path), 'Model weights not found.'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        
    f.close()
    print('Model loaded.')

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    
    if is_test == False:
        bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    else:    
        bottleneck_features_train = model.predict_generator(generator, nb_test_samples)

    
    if mode == "Evaluate":
        generator = datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=32,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    
    if save_model == True:
        np.save(bottleneck_features_train_filename, bottleneck_features_train)
        
        if mode == "Evaluate":
            np.save(bottleneck_features_validation_filename, bottleneck_features_validation)

        print("Saved bottleneck_features_validation.")
    
    print("Saved bottleneck_features_train.")
    print("Finish generating bottleneck_features.")


def train_top_model(top_model_weights_path,
                    bottleneck_features_train_filename,
                    bottleneck_features_validation_filename,
                    labels_train,
                    labels_validation, nb_epoch):
    
    train_data = np.load(bottleneck_features_train_filename)
    train_labels = labels_train

    if mode == "Evaluate":
        validation_data = np.load(bottleneck_features_validation_filename)
        validation_labels = labels_validation

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(40, activation='softmax'))

    print("Define the newly-added layers.")

    sgd = Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Start training the model.")
    
    if mode == "Evaluate":
        model.fit(train_data, train_labels,
                  nb_epoch=nb_epoch, batch_size=32,
                  validation_data=(validation_data, validation_labels))
    else:
        model.fit(train_data, train_labels,
                  nb_epoch=nb_epoch, batch_size=32)
    
    model.save_weights(top_model_weights_path, overwrite = True)
    
    return model
    
    
print "Done..............."


# In[ ]:

print("\nBegin extracting features from the pretrained model.")


bottleneck_features_train_filename = root_folder_name + 'bottleneck_features_train.npy'
bottleneck_features_validation_filename = root_folder_name + 'bottleneck_features_validation.npy'

bottleneck_features_train_filename_for_prediction = root_folder_name_for_prediction + 'bottleneck_features_train_for_prediction.npy'
bottleneck_features_validation_filename_for_prediction = root_folder_name_for_prediction + 'bottleneck_features_validation_for_prediction.npy'
bottleneck_features_test_filename_for_prediction = root_folder_name_for_prediction + 'bottleneck_features_test_for_prediction.npy'


print("\nRecord the time")
start_time = time.time()

if calculate_features == True:
    if mode == "Evaluate":
        save_bottleneck_features(train_data_dir,
                                 validation_data_dir,
                                 weights_path,
                                 bottleneck_features_train_filename,
                                 bottleneck_features_validation_filename,
                                 img_width,
                                 img_height, save_model)
    else:
        save_bottleneck_features(train_data_dir_for_prediction,
                                 validation_data_dir_for_prediction,
                                 weights_path,
                                 bottleneck_features_train_filename_for_prediction,
                                 bottleneck_features_validation_filename_for_prediction,
                                 img_width,
                                 img_height, save_model)

end_time = time.time()
print("--- %s seconds ---" % (end_time - start_time))

print "Done........"


# In[ ]:

print("\nTrain the newly added FC layer.")

print("Record the time.\n")
start_time = time.time()

if mode == "Evaluate":
    model = train_top_model(top_model_weights_path,
                    bottleneck_features_train_filename,
                    bottleneck_features_validation_filename,
                    labels_train,
                    labels_val,
                    nb_epoch)

else:
    model = train_top_model(top_model_weights_path_for_prediction,
                            bottleneck_features_train_filename_for_prediction,
                            bottleneck_features_validation_filename_for_prediction,
                            labels_train,
                            labels_val,
                            nb_epoch)

end_time = time.time()
print("\n--- %s seconds ---" % (end_time - start_time))


# In[51]:

model.summary()


# In[15]:

if mode == "Predict":

    if calculate_features_for_prediction == True:
        save_bottleneck_features(test_data_dir_for_prediction,
                                 None,
                                 weights_path,
                                 bottleneck_features_test_filename_for_prediction,
                                 None,
                                 img_width,
                                 img_height,
                                 save_model,
                                 is_test = True)
        
    print("Making the predictions.")
    print("Record the time")
    start_time = time.time()
    
    test_data_features = np.load(bottleneck_features_test_filename_for_prediction)
    prediction_results = model.predict(test_data_features)   
    prediction = prediction_results.argmax(axis = 1)

    with open("../predictions_weights/"+test_predict_CNN_filename + time.strftime("%d_%m_%Y_%H_%M_%S") + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'class'])

        for i in range(prediction.shape[0]):

            writer.writerow([i, prediction[i]])


    end_time = time.time()
    print("Written CNN prediction result to the file!")
    print("--- %s seconds ---" % (end_time - start_time))


# In[16]:

plt.hist(prediction, bins=40)
plt.title("Prediction Distribution")
plt.show()


# In[57]:

# %matplotlib inline
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
loss = plt.plot(model.model.history.history['loss'],'r')
acc = plt.plot(model.model.history.history['acc'],'b')
val_loss = plt.plot(model.model.history.history['val_loss'],'g')
val_acc = plt.plot(model.model.history.history['val_acc'],color="yellow")
# plt.legend([loss, acc, val_loss,val_acc], ['1','2','3','4'])
red_patch = mpatches.Patch(color='red', label='Training Loss')
blue_patch = mpatches.Patch(color='blue', label='Training Accuracy')
green_patch = mpatches.Patch(color='green', label='Validation Loss')
yellow_patch = mpatches.Patch(color='yellow', label='Validation Accuracy')
plt.legend(handles=[red_patch,blue_patch,green_patch,yellow_patch])
plt.title("Training And Validation Loss and Accuracy")
plt.xlabel("No. of Epochs")
plt.ylabel("Loss and Accuracy")
plt.show()


