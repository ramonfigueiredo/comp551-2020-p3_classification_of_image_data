import os
import numpy
import scipy
import scipy.misc
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import random, string


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))
OUTPUT_DIR = "data/label"
trainX = numpy.load('../data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = numpy.load('../data/tinyY.npy') 
testX = numpy.load('../data/tinyX_test.npy') # (6600, 3, 64, 64)

if not os.path.isdir("data"):
    os.makedirs("data")

labels = ["%d" % label for label in trainY]
#
#for i in range(len(trainX)):
#    label = labels[i]
#    new_path = OUTPUT_DIR + label
#    if not os.path.isdir(new_path):
#        os.makedirs(new_path)
#    
#    x= trainX[i]
#    x = x.reshape((1,) + x.shape)
#
#    i = 0
#    for batch in datagen.flow(x, batch_size=1,
#                          save_to_dir=OUTPUT_DIR + label + "/", save_prefix=randomword(32), save_format='jpeg'):
#        i += 1
#        if i > 20:
#            break  # otherwise the generator would loop indefinitely
        
#        im = Image.fromarray(trainX[i].transpose(2,1,0))
#    im.save(OUTPUT_DIR + label + "/" + randomword(32) + ".jpeg")

for i in range(len(trainX)):
    label = labels[i]
    new_path = OUTPUT_DIR + label
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    im = Image.fromarray(trainX[i].transpose(2,1,0))
    im.save(OUTPUT_DIR + label + "/" + randomword(32) + ".jpeg")