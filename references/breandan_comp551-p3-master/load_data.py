'''
Loads training data, presumed to be in ./data
'''

import numpy as np
from pathlib import Path
from collections import defaultdict

def load_testing_data():
    path = "data/testing_data.npz"
    testing_data = Path(path)
    if testing_data.is_file():
        with np.load(path) as data:
            x = data['x']
    else:
        x = np.genfromtxt('data/test_x.csv', delimiter=",")
        np.savez(path, x=x)

    return np.uint8(x.reshape(-1, 64, 64, 1))

def load_training_data():
    path = "data/training_data.npz"
    training_data = Path(path)
    
    if training_data.is_file():
        with np.load(path) as data:
            x = data['x']
            y = data['y']
    else:
        x = np.genfromtxt('data/train_x.csv', delimiter=",")
        y = np.genfromtxt('data/train_y.csv', dtype=int)
        np.savez(path, x=x, y=y)

    #y = [transform(yi) for yi in y]
    #x_copy = []
    #y_copy = []
    #
    #for i, yi in enumerate(y):
    #    if yi != None:
    #        x_copy.append(x[i])
    #        y_copy.append(yi)

    return np.uint8(np.array(x)), np.array(y)

def get_labels():
    numbers = set()
    solutions = defaultdict(int)
    for i in range(0,10):
        for j in range(0,10):
            numbers.add(i+j)
            numbers.add(i*j)
            solutions[i+j] += 1
            solutions[i*j] += 1
    uniques = {k:v in solutions for (k, v) in solutions.items() if v == 2}

    mapping = [-1]*82
    for i, j in enumerate(numbers):
       mapping[j] = i 

    return numbers, mapping, uniques

def transform(number):
    d = {17:['a', 7, 9],
         20:['m', 4, 5],
         21:['m', 3, 7],
         25:['m', 5, 5],
         27:['m', 3, 9],
         28:['m', 4, 7],
         30:['m', 5, 6],
         32:['m', 4, 8],
         35:['m', 5, 7],
         40:['m', 5, 8],
         42:['m', 6, 7],
         45:['m', 5, 9],
         48:['m', 6, 8],
         49:['m', 7, 7],
         54:['m', 6, 9],
         56:['m', 7, 8],
         63:['m', 7, 9],
         64:['m', 8, 8],
         72:['m', 8, 9],
         81:['m', 9, 9]}

    template = [0]*12

    if number not in d:
        return None

    for x in d[number]:
        if x == 'm':
            template[10] = 1
        elif x == 'a':
            template[11] = 1
        else:
            template[x] = 1

    return template
