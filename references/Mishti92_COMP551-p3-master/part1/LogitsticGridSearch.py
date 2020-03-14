
# coding: utf-8

# ## Baseline Learner
# logistic regression

# In[25]:

import numpy as np
import argparse
import cv2
from math import sqrt
from sklearn.decomposition import PCA

def adjust_gamma(image, gamma=2.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def flattenImg(X):
    # take every image into row vector
    n_samples, n_dim1, n_dim2 = X.shape
    x_flatten = np.zeros((n_samples, n_dim1 * n_dim2))
    
    for i in range(n_samples):
        x_flatten[i] = X[i].flatten()
    return x_flatten

def transBack(X):
    '''
    input: flattend X, where every row is a flattend image/
    output: transfrom X back to image square matrix
    '''
    print X.shape
    n_samples, n_dim = X.shape
    dim = int(sqrt(n_dim))
    
    x_trans = np.zeros((n_samples, dim, dim))
     
    for i in range(n_samples):
        x_trans[i] = X[i].reshape((dim,dim))
    return x_trans
    
def preprocessing(train_x, test_x, dim = 20):
    '''
    input: raw train_x, dim: target dimension for PCA
    '''
    # gamma correction on image
    train_x = adjust_gamma(train_x, gamma=2.2)
    # RGB to greyscale
    train_x = np.mean(train_x, axis = 1)
    # mean subtraction
    train_x = train_x - np.mean(train_x)
    # normalization
    train_x = train_x/np.std(train_x)
    
    # gamma correction on image
    test_x = adjust_gamma(test_x, gamma=2.2)
    # RGB to greyscale
    test_x = np.mean(test_x, axis = 1)
    # mean subtraction
    test_x = test_x - np.mean(test_x)
    # normalization
    test_x = test_x/np.std(test_x)
    
    # convert image square matrix into a row
    # comment this line if you want original format (square matrix per image) 
    train_x = flattenImg(train_x)
    test_x = flattenImg(test_x)  
    # ----  apply PCA to reduce dimension----
    n_samples, n_dim1 = train_x.shape
    pca = PCA(n_components=dim)
    pca.fit(train_x)
    train_x = pca.transform(train_x)
    test_x = pca.transform(test_x)

    return train_x, test_x




# In[26]:

import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,  fontsize = 5)
    plt.yticks(tick_marks, classes,fontsize = 5)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:




# In[27]:

import itertools
from sklearn import  linear_model
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

train_x = np.load("../Data/tinyX.npy")
y = np.load("../Data/tinyY.npy")
test_x = np.load("../Data/tinyX_test.npy")

X,test_x_reduced = preprocessing(train_x, test_x,dim = 2048)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=0)

# logistic = linear_model.LogisticRegression()
# logistic.fit(X_train, y_train)
# y_true, y_pred = y_test, logistic.predict(X_test)
# print classification_report(y_true, y_pred)

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)

# # Plot normalized confusion matrix
# plt.figure()
# class_names = range(40)
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()

tuned_parameters = [{'penalty': ['l1', 'l2'],
                    'C': [1, 10, 100, 1000],
                   'class_weight' :['balanced'],
                    'multi_class' :['multinomial','ovr'],
                     'max_iter':[1000] }, ]

# tuned_parameters = [{'penalty': ['l1'],
#                     'C': [1],
#                     'verbose':[1]}]
logreg = linear_model.LogisticRegression()
clf = GridSearchCV(logreg, tuned_parameters)
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)
print classification_report(y_true, y_pred), clf.score(X_test, y_test)
print clf.best_estimator_.get_params()

# Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)

# Plot normalized confusion matrix
# fig = plt.figure()
# class_names = range(40)
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()
# fig.savefig('../data/confusion.png')

# print clf.best_score_

# y_pred = clf.predict_proba(X_test)
# y_score = np.max(y_pred, axis=1)
# fpr, tpr, th = roc_curve(y_test, y_score)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.plot(fpr, tpr)
# plt.savefig('roc.png')


# In[7]:




# In[ ]:




# In[ ]:



