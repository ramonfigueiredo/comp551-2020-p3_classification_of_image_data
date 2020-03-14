'''Cleans data from the COMP 551 Kaggle dataset.
'''

import cv2
import numpy as np
from load_data import load_training_data
from load_data import get_labels
import scipy.misc

def get_clean_data():
    x, y = load_training_data()

    # Reshape images from 1x4096 to 64x64
    x = threshold(x)
    x = remove_dots(x)
    x = x.reshape(-1, 64, 64, 1)

    test_split = 0.1
    np.random.seed(113)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    _, num_to_index, _ = get_labels()
    x = x[indices]
    y = y[indices]
    y = [num_to_index[yi] for yi in y.tolist()]

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)

def threshold(x):
    # Binarize input data
    x[x > 200] = 255
    x[x < 255] = 0

    return x

# Clean up noise
def remove_dots(x):
    #x = x.reshape(-1, 64, 64)
    for j, img in enumerate(x):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # Minimum number of connected pixels to keep
        min_size = 20

        img2 = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
        x[j] = img2.reshape(64, 64)

    return x
