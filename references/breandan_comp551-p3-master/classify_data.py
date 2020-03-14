from tensorflow.python.keras.models import load_model
from load_data import load_testing_data, get_labels
from clean_data import threshold, remove_dots
import scipy.misc, scipy.ndimage
import numpy as np
import sys

if len(sys.argv) < 2:
    model_path = 'data/temp_model.hdf5'
else:
    model_path = sys.argv[1]

if len(sys.argv) < 3:
    output_path = 'data/test_y.csv'
else:
    output_path = sys.argv[2]

print("Loading model from " + model_path + "...")
model = load_model(model_path)

print("Loading test data...")
x_test = load_testing_data()
x_test = x_test.reshape(-1, 64, 64)

print("Removing background...")
x_test = threshold(x_test)

print("Removing dots...")
#x_test = remove_dots(x_test)

#new_im = np.zeros((256, 256))
#r = 0
#for i in range(0, 256,64):
#    for j in range(0, 256, 64):
#        # paste the image at location i,j
#        new_im[i:i+64, j:j+64] = x_test[r]
#        r += 1
#
#new_im = scipy.ndimage.zoom(new_im, 4, order=0) 
#scipy.misc.imsave('sample_clean.jpg', new_im)

#scipy.misc.imshow(new_im) # to visualize only 
#quit()

print("Generating predictions...")
x_test = x_test.reshape(-1, 64, 64, 1)
predictions = model.predict(x_test)

labels, _, _ = get_labels()
labels = list(labels)

print("Writing predictions to " + output_path + "...")
with open(output_path, 'w+') as output:
    print("Id,Label", file=output)
    for i, prediction in enumerate(predictions):
        print(str(i + 1) + "," + str(labels[np.argmax(prediction)]), file=output)
