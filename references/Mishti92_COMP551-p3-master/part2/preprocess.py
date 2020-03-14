from sklearn import datasets
import numpy as np
import sklearn

import ann
import ann1
import logistic 

SPLIT = 0.7
train_x = np.load("../data/tinyX.npy")
Y = np.load("../data/tinyY.npy")
test_x = np.load("../data/tinyX_test.npy")

dimn = 200
X, XTest = logistic.preprocessing(train_x, test_x,dim = dimn)
np.save("../data/X_"+str(dimn)+".npy",X)
np.save("../data/XTest_"+str(dimn)+".npy",XTest)
print "Completed dim: ", dimn 

dimn = 500
X, XTest = logistic.preprocessing(train_x, test_x,dim = dimn)
np.save("../data/X_"+str(dimn)+".npy",X)
np.save("../data/XTest_"+str(dimn)+".npy",XTest)
print "Completed dim: ", dimn

dimn = 1000
X, XTest = logistic.preprocessing(train_x, test_x,dim = dimn)
np.save("../data/X_"+str(dimn)+".npy",X)
np.save("../data/XTest_"+str(dimn)+".npy",XTest)
print "Completed dim: ", dimn

dimn = 1500
X, XTest = logistic.preprocessing(train_x, test_x,dim = dimn)
np.save("../data/X_"+str(dimn)+".npy",X)
np.save("../data/XTest_"+str(dimn)+".npy",XTest)
print "Completed dim: ", dimn

dimn = 2000
X, XTest = logistic.preprocessing(train_x, test_x,dim = dimn)
np.save("../data/X_"+str(dimn)+".npy",X)
np.save("../data/XTest_"+str(dimn)+".npy",XTest)
print "Completed dim: ", dimn
