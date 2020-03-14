import numpy as np 
import scipy.misc # to visualize only 
import csv
from PIL import Image
from PIL import ImageOps
import sys

#globals parameters for function
size = 16
answer = dict()


#turns the image into a resize better formatted string to be used for our algorithm
def getImageString(pixels, answerIndex):
    string = ""
    x = np.loadtxt(pixels, delimiter=",")
    x = x.reshape(-1, 64, 64) # reshape
    x = x/-255.0
    img = Image.fromarray(np.uint8(x[0]),'L')
    img.thumbnail([size,size],Image.ANTIALIAS)
    img = ImageOps.crop(img,int(size/10))
   
    for pixel in iter(img.getdata()):
        string += str(pixel)+","
    string += str(answer[answerIndex])+"\n"
    return string

def main():
    #open files to be used
    validationFile = open("../"+str(size)+"validation_set.csv","w")
    training_set = open("../"+str(size)+"training_set.csv","w")
    test_data = open("../"+str(size)+"testing_set.csv","w")
    testFile = open("../"+str(size)+"test_set.csv","w")

    print "getting answers \n"
    i = 1
    for row in csv.reader(open("../train_y.csv")):
        answer[i] = int(row[0])
        i += 1
    print "finished getting answers\n starting creation of files"

    #make the training and validation in the right size
    i = 1
    j = 0
    for row in csv.reader(open("../train_x.csv")):
        if i%5 == 0:  
            #swap for making 2 5k set and one 40k set
            if j == 0:
                validationFile.write(getImageString(row,i))
                j = 1
            else:
                test_data.write(getImageString(row,i))
                j = 0
        else:
            training_set.write(getImageString(row,i))
        print str(i/50000.0*100) +" "+ str(i) +"/50000 %completed"
        i += 1

    i = 1
    for row in csv.reader(open('../test_x.csv')):
        testFile.write(getImageString(row,i).rstrip(",")+"\n") 
        i += 1
        print str(i/10000.0*100) +" "+ str(i) +"/10000 %completed"
    print "Closing Files"
    training_set.close()
    validationFile.close()
    testFile.close()



        
      
if __name__ == "__main__":
    main()
