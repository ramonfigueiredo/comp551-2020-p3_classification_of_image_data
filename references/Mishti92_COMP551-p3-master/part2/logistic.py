
# coding: utf-8

# ## Baseline Learner
# logistic regression

# In[26]:

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




# In[ ]:




# In[24]:


# from sklearn import  linear_model

#train_x = np.load("../Data/tinyX.npy")
#train_y = np.load("../Data/tinyY.npy")
#test_x = np.load("../Data/tinyX_test.npy")

#train_x_reduced,test_x_reduced = preprocessing(train_x, test_x,dim = 50)

#logistic = linear_model.LogisticRegression()
#logistic.fit(train_x, train_y)
#predict_y = logistic.predict(test_x)
#np.save('predict_y.npy', x)


# In[20]:




# In[ ]:




# In[ ]:



