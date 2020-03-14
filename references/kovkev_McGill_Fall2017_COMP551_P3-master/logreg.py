from __future__ import division
import numpy as np
import time
import csv
from sklearn import datasets, linear_model

train_x = []
train_y = []
test_x = []
with open("train_x.csv") as csvfile:
    reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = np.array(row).astype(float).astype(np.uint8)
        row[row>240] = 1
        row[row<=240] = 0
        train_x.append(row)

with open("train_y.csv") as csvfile:
    reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = np.array(row).astype(float).astype(np.uint8)
        train_y.append(row)

with open("test_x.csv") as csvfile:
    reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = np.array(row).astype(float).astype(np.uint8)
        row[row>240] = 1
        row[row<=240] = 0
        test_x.append(row)


regr = linear_model.LogisticRegression()
train_y = np.array(train_y)
train_y = train_y.ravel()
print "fitting the model"
print time.clock()
regr.fit(train_x, train_y)
print time.clock()
print "model training complete"

# validation = regr.predict(train_x[40000:50000])
#
# counter = 0;
# for x,y in zip(validation, train_y[40000:50000]):
#     if x == y:
#         counter += 1
# print ("counter is: %s" % counter)
# validationaccuracy = counter/len(validation)
# print ("validation accuracy: %s" % validationaccuracy)

test_y = regr.predict(test_x)

file = open('test_y.csv', 'w')
file.write("Id,label\n")
counter = 0
for output in test_y:
   counter += 1
   file.write("%s,%s\n" % (counter, output))
file.close()
