import os
import numpy
import scipy
import scipy.misc
from PIL import Image
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import random, string



testX = numpy.load('../data/tinyX_test.npy') # (6600, 3, 64, 64)

if not os.path.isdir("test"):
    os.makedirs("test")


for i in range(len(testX)):

    label = str(i)
    path = "test/" + label + ".jpeg" 

    im = Image.fromarray(testX[i].transpose(2,1,0))
    im.save(path)