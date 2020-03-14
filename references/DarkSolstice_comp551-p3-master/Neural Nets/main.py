from collections import defaultdict
import csv
from StringIO import StringIO
import sys
import math
import numpy as np 
import Network
import NeuralNet
from PIL import Image
from PIL import ImageOps

valueList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
possibleOutcomes = 40

#turns the result into a matrix of N, 1 where the value is 1 in the jth column
def vectorized_result(j):
    #turn the value into an int between 0 and the number of outcomes to remove output nodes that wont be used
    index = valueList.index(j);
    e = np.zeros((possibleOutcomes, 1))
    e[index] = 1.0
    return e

def main():
    #original were resized to 32x32 and cropped to 26x26
    size = 26
    #cropped by about 10% on each side
    cropped = 3
    #filtering amount for each type set high if you want the whole data set
    countOfEach = 150000
    #get every value until 40 = 81 because its the highest index

    #initialize the network class with 5 layers need more for more accuracy or else it defaults to selecting one answer only
    print "start "

    
    trainingFile = "../"+str(size+cropped*2)+"training_set.csv"
    validationFile = "../"+str(size+cropped*2)+"validation_set.csv"
    testingFile = "../"+str(size+cropped*2)+"testing_set.csv"
    testFile = "../"+str(size+cropped*2)+"test_set.csv"

    training_x = []
    training_y = []
    i = 0;
    #filtering results to even out the amount of each type even tho the test set doesnt have all equal amounts
    #for the learning process this is crucial
    about = dict()
    for row in csv.reader(open(trainingFile)):
        y = valueList.index(int(row[size*size]))
        if y in about:
            about[y] += 1
        else:
            about[y] = 1
        if about[y] < countOfEach:                   
            training_y.append(vectorized_result(int(row[size*size])).tolist())
            del row[size*size]
            floats = [float(pixel) for pixel in row]
            training_x.append(list(floats))
        

    training_x = [np.reshape(x, (size*size, 1)) for x in training_x]
    training_y = np.array(training_y)
    training_data = zip(training_x, training_y)

    validation_x = []
    validation_y = []
    for row in csv.reader(open(validationFile)):
        validation_y.append(valueList.index(int(row[size*size])))
        del row[size*size]
        floats = [float(pixel) for pixel in row]
        validation_x.append(list(floats))

    validation_x = [np.reshape(x, (size*size, 1)) for x in validation_x]
    validation_y = np.array(validation_y)      
    validation_data = zip(validation_x, validation_y)

    test_y = []
    test_x = []
    for row in csv.reader(open(testingFile)):
        test_y.append(valueList.index(int(row[size*size])))
        #test_y.append(int(row[size*size]))
        del row[size*size]
        floats = [float(pixel) for pixel in row]
        test_x.append(list(floats))

    test_x = [np.reshape(x, (size*size, 1)) for x in test_x]
    test_y = np.array(test_y)      
    test_data = zip(test_x, test_y)


    print "starting network test\n"  
    print "After filterting training_data has: "+str(len(training_data))+" inputs.\n"

    net = NeuralNet.NeuralNet([size*size,200,100,possibleOutcomes])
    #start stochastic gradient descent learning with epoch 30 and learning rate of 3.0 with batch sizes of 40
    net.SGD(training_data, 40, 10, 3.0, test_data=test_data)
    print str(size) + "x" + str(size) + " with layers: "+str(size*size)+" 200,100,"+str(possibleOutcomes)
    net = NeuralNet.NeuralNet([size*size,200,possibleOutcomes])
    net.SGD(training_data, 40, 10, 3.0, test_data=test_data)
    print str(size) + "x" + str(size) + " with layers: "+str(size*size)+" 200,"+str(possibleOutcomes)
    net = NeuralNet.NeuralNet([size*size,300,150,possibleOutcomes])
    net.SGD(training_data, 40, 10, 3.0, test_data=test_data)
    print str(size) + "x" + str(size) + " with layers: "+str(size*size)+" 200,100,"+str(possibleOutcomes)
      
if __name__ == "__main__":
    main()
