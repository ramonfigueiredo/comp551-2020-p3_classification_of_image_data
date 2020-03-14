from collections import defaultdict
import csv
from StringIO import StringIO
import sys
import math
import numpy as np 
from PIL import Image
from PIL import ImageOps
from sklearn import svm

valueList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
    
def main():
    #original were resized to 32x32 and cropped to 26x26
    size = 14
    #cropped by about 10% on each side
    cropped = 1
    #filtering amount for each type set high if you want the whole data set
    countOfEach = 100
    
    trainingFile = "../"+str(size+cropped*2)+"training_set.csv"
    validationFile = "../"+str(size+cropped*2)+"validation_set.csv"
    testingFile = "../"+str(size+cropped*2)+"testing_set.csv"
    testFile = "../"+str(size+cropped*2)+"test_set.csv"

    training_x = []
    training_y = []
    i = 0
    amount = dict();
    #filtering results to even out the amount of each type even tho the test set doesnt have all equal amounts
    #for the learning process this is crucial
    for row in csv.reader(open(trainingFile)):         
            y = valueList.index(int(row[size*size]))  
            if y in amount:
                amount[y] += 1
            else:
                amount[y] = 1
            if amount[y] <= countOfEach:      
                training_y.append(y)
                del row[size*size]
                floats = [float(pixel) for pixel in row]
                training_x.append(list(floats))
        
        

    training_x = [np.asarray(x, dtype=np.float64) for x in training_x]
    training_y = np.array(training_y)
    #training_data = zip(training_x, training_y)
    print len(training_x)
    print "Starting fitting"
    clf = svm.SVC(kernel="rbf",cache_size=3000,C=50.0,gamma=2.0,coef0=1.0,decision_function_shape='ovr')
    clf.fit(training_x, training_y) 
    predictions = clf.predict(training_x)
    print str(sum(int(x == y) for (x, y) in zip(predictions,training_y))) + "/" + str(len(predictions))
    
    validation_x = []
    validation_y = []
    for row in csv.reader(open(validationFile)):
        validation_y.append(valueList.index(int(row[size*size])))
        del row[size*size]
        floats = [float(pixel) for pixel in row]
        validation_x.append(list(floats))

    validation_x = [np.asarray(x, dtype=np.float64) for x in validation_x]
    validation_y = np.array(validation_y)      
    #validation_data = zip(validation_x, validation_y)
    predictions = clf.predict(validation_x)

    print str(sum(int(x == y) for (x, y) in zip(predictions,validation_y))) + "/" + str(len(predictions))
    sys.exit()
      
if __name__ == "__main__":
    main()
