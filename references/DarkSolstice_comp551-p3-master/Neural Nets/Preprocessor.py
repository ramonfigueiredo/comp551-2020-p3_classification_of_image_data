from collections import defaultdict
import csv
from StringIO import StringIO
import sys
import math
from datetime import datetime
import numpy as np 
import scipy.misc # to visualize only 
import Network


input_x_filename = "../train_x.csv"
input_y_filename = "../train_y.csv"
test_set_filename = "../test_x.csv"

    
def main():
    answer = dict()
    i = 1
    for row in csv.reader(open(input_y_filename)):
        answer[i] = int(row[0])
        i += 1
    i = 1
    print "starting the filewritting"
    validationFile = open("../validation_set.csv","w")
    training_set = open("../training_set.csv","w")
    for row in csv.reader(open(input_x_filename)):
        toWrite = ""
        for value in row:
            toWrite += str(int(float(value)))+","               
        toWrite += str(answer[i])+"\n"
        if i%10 == 0:
            validationFile.write(toWrite)
        else:
            training_set.write(toWrite)
        print str(i/50000.0*100) +"% completed"
        i = i + 1

    print "Closing Files"
    training_set.close()
    validationFile.close()
        
      
if __name__ == "__main__":
    main()
