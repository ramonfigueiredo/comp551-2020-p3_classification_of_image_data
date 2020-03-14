print "#######################################################################"
import numpy as np
import sklearn

import ann
#import logistic 
print "#######################################################################"
SPLIT = 0.8

dimn = 2000
X = np.load("../data/X_"+str(dimn)+".npy")
Y = np.load("../data/tinyY.npy")
XTest = np.load("../data/XTest_"+str(dimn)+".npy")

print "#1debug: ", X.shape

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
XTest = (XTest - np.mean(XTest, axis=0)) / np.std(XTest, axis=0)

m,n = X.shape
lookupTable, Y = np.unique(Y, return_inverse=True)
X, Y = sklearn.utils.shuffle(X, Y)
index = int(SPLIT * m)
trainX = X[:index][:]
trainY = Y[:index]
cvX = X[index:][:]
cvY = Y[index:]

inputs = n
outputs = len(lookupTable)
nodes_layer1 = 100
learning_rate = 0.8
reg_param = 1e-2
threshold_fn = 'logistic'
cost_fn = 'cross_entropy'
epochs = 500
momentum_rate = 0.8
learning_accelaration = 1.05
learning_backup = 0.5

dimensions = [200,500,1000,1500,2000]
results = []
for i in range(len(dimensions)):
    dimn = dimensions[i]
    X = np.load("../data/X_"+str(dimn)+".npy")
    Y = np.load("../data/tinyY.npy")
    XTest = np.load("../data/XTest_"+str(dimn)+".npy")

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    XTest = (XTest - np.mean(XTest, axis=0)) / np.std(XTest, axis=0)

    m,n = X.shape
    lookupTable, Y = np.unique(Y, return_inverse=True)
    X, Y = sklearn.utils.shuffle(X, Y)
    index = int(SPLIT * m)
    trainX = X[:index][:]
    trainY = Y[:index]
    cvX = X[index:][:]
    cvY = Y[index:]

    inputs = n
    outputs = len(lookupTable)
    
    learning_rate = 1e-2
    a = ann.ANN(inputs, outputs, [nodes_layer1], epochs, learning_rate, momentum_rate, learning_accelaration, learning_backup, reg_param, threshold_fn, cost_fn)
    a.fit(trainX, trainY)
    predY = a.predict(cvX)
    #predY1 = a.predict(XTest)
    
    from sklearn.metrics import accuracy_score
    a = accuracy_score(cvY, predY)
    results.append((dimn, a))
    print "debug#5- dimension: ", dimn, " , accuracy: ", a

results = np.array(results)
np.save('results5.npy', results)
