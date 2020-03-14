from __future__ import division

import numpy as np
from sklearn import utils

class ANN:
    
    def __init__(self, inputs, outputs, hidden_layers, epochs, learning_rate, momentum_rate, learning_accelaration, learning_backup, reg_param, activation_fn, cost_fn):

        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.learning_accelaration = learning_accelaration
        self.learning_backup = learning_backup
        self.reg_param = reg_param        
        self.epochs = epochs
        self.outputs = outputs
 
        if activation_fn == "tanh":
            self.activation_fn = self.__tanh
            self.gactivation_fn = self.__gtanh
        elif activation_fn == "logistic":
            self.activation_fn = self.__logistic
            self.gactivation_fn = self.__glogistic

        if cost_fn == "cross_entropy":
            self.cost_fn = self.__cross_entropy
        elif cost_fn == "least_squares":
            self.cost_fn = self.__least_squares

        self.weights = []
        ul = [inputs] + hidden_layers + [outputs]
        for l,u in enumerate(ul):
            if l > 0:
                w = 2 * (np.random.rand( u, ul[l-1]+1 ) - 0.5)
                #w = np.random.rand( u, ul[l-1]+1 )
                self.weights.append(w)

    def sample_training_data(self,X,Y,percent):
        X, Y = utils.shuffle(X, Y)
        m,_ = X.shape
        return (X[:int(percent*m)][:],Y[:int(percent*m)][:])

    def fit(self, X, Y):
        #X = X / X.max(axis=0)
        SAMPLE_SIZE = 1.0
        Y = self.__get_yarray((len(Y), self.outputs), Y)

        weight_changes = [np.zeros_like(theta) for theta in self.weights]
        
        (Xmini, Ymini) = self.sample_training_data(X, Y, SAMPLE_SIZE)
        #Xmini = X
        #Ymini = Y
        YPred = self.forward_propagate(Xmini)

        Err0 = self.cost_fn(Ymini, YPred) 
        
        for k in range(self.epochs):
            #(Xmini, Ymini) = self.sample_training_data(X, Y, SAMPLE_SIZE)
            #Ymini = self.__get_yarray(YPred.shape, Ymini)
            
            self.backward_propagate(Ymini)
            for i, dW in enumerate(self.gweights):
                weight_changes[i] = self.learning_rate * dW + (weight_changes[i] * self.momentum_rate)
                #print "debug #1: ", weight_changes[i]
                self.weights[i] -= weight_changes[i]

            YPred = self.forward_propagate(Xmini)
            Err1 = self.cost_fn(Ymini, YPred)

            #(Xmini, Ymini) = self.sample_training_data(X, Y, SAMPLE_SIZE)
 
            if Err1 > Err0:
                self.learning_rate = self.learning_rate * self.learning_backup
                self.weights = [t + (self.learning_rate * tg) \
                    for t, tg in zip(self.weights, self.gweights)]
                weight_changes = [m * self.learning_backup for m in weight_changes]

                YPred = self.forward_propagate(Xmini)
                Err1 = self.cost_fn(Ymini, YPred)
            else:
                self.learning_rate = np.min((10,self.learning_rate * self.learning_accelaration))

            Err0 = Err1
            #print "debug: iteration#: ", k, " Error: ", Err1
 
        return None

    def backward_propagate(self, Y):
        n,_ = Y.shape
        deltas = []
        delta = self.activations[-1] - Y
        deltas.append(delta)
        
        T = len(self.weights)
        for t in range(T-1,0,-1):
            delta = delta.dot(self.weights[t][:,1:]) * (self.gactivation_fn(self.activations[t][:,1:]))
            deltas.append(delta)
        deltas.reverse()

        self.gweights = []

        for t in range(T):
            self.gweights.append(deltas[t].T.dot(self.activations[t]))
        
        reg = [np.zeros_like(theta) for theta in self.weights]
        for t, theta in enumerate(self.weights):
            reg[t][:,1:] = theta[:,1:]

        for t in range(T):
            self.gweights[t] = self.gweights[t] * (1.0/n) + (self.reg_param * reg[t])

        return None

    def forward_propagate(self, X):
        self.activations = []
        
        self.activations.append(X)

        for t, w in enumerate(self.weights):
            if self.activations[t].ndim == 1:
                self.activations[t].resize(1, self.activations[t].shape[0])
            self.activations[t] = np.append(np.ones((self.activations[t].shape[0],1)), self.activations[t], 1)
            z = self.activations[t].dot(w.T)
            a = self.activation_fn(z)
            self.activations.append(a)
        return a

    def predict(self, X):
        a = self.forward_propagate(X)
        exp_scores = np.exp(a)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def __cross_entropy(self, Y, YPred):
        #Y = self.__get_yarray((len(Y),3), Y)
        #YPred = self.__get_yarray(Y.shape,YPred)
        m = len(Y)
        d = 1e-6
    # Cost Function
        Err = (Y * np.log(YPred + d) + ((1-Y) * np.log(1-YPred + d))).sum()
        return (-Err/m)

    def __get_yarray(self,size,Y):
        (m,_) = size
        arr = np.zeros(size)
        for i in range(m):
            arr[i,Y[i]] = 1
        return arr

    def __least_squares(self, Y, YPred):
        return np.sum(np.sum(np.power((Y - YPred), 2)))

    def __stochastic_gd(self):
        pass
    
    def __minibatch_gd(self):
        pass

    def __batch_gd(self):
        pass

    def __tanh(self, X):
        return np.tanh(X)

    def __gtanh(self, X):
        return 1 - np.power(X,2)

    def __logistic(self, X):
        return 1 / (1 + np.exp(-X)) 

    def __glogistic(self, X):
        return np.multiply(X, 1-X)

