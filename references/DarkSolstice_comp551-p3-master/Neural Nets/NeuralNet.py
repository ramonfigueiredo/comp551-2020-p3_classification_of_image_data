"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import sys
# Third-party libraries
import numpy as np

class NeuralNet(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #biases for layers 2 to n
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #weights for layers 2 to n
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        

    def feedforward(self, a):
        #w is matrix of weigths of size layer layer 2 * layer 1
        for b, w in zip(self.biases, self.weights):
            a = activationFunction(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        #times tried training
        for j in xrange(epochs):
            #randomizes the data so that we dont always take the same one
            random.shuffle(training_data)
            #creates subsets of size mini batch size from position k so if total points = 50000 it will create 50000/mini_batch_size
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            #for each data point in the batch
            for mini_batch in mini_batches:
                #update weights for each batch
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        #create temporary arrays for new weights and biases changes to be generated
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #substract part of the adjustment bases on size of batch because if doing by 10 they should have all 1/10 of the impact
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            #multiply each weight by its input
            z = np.dot(w, activation)+b
            zs.append(z)
            #return sigmoid of each 
            activation = activationFunction(z)
            activations.append(activation)
        #after this loop we did a fully connected calcualtion of layer n->n+1
        #the results from each layer are in each index of activations starting at 0

        # backward pass
        #starting from the last layer we calculate the 
        #heres the important sum of error * derivative of sigmoid
        delta = self.cost_derivative(activations[-1], y) * activationFunction_d(zs[-1])

        #set the new biases needed
        nabla_b[-1] = delta
        transpose = activations[-2].transpose()
        #creating new weights by updating them with the previous layers weights and the new biases
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())


        #propagate the changes through out the neural net starting from the before last layer and going backwards
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = activationFunction_d(z)
            #grabbing the vector from the calculations above and multiplying with the old weights delta of n+1 layer
            #create new delta and update new weights like before
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        
        #for each x in test_data run the neural net and pair with its correct answer
        timesGuessed = dict()
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        for (x,y) in test_results:
            if x in timesGuessed:
                timesGuessed[x] += 1
            else:
                timesGuessed[x] = 1
        print timesGuessed
        #sum how many have the correct answer             
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations-y

#these are just wrapper to easily change the activation function while testing
def activationFunction(z):
    return sigmoid(z)
def activationFunction_d(z):
    return sigmoid_prime(z)
#### activation function types
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def arctan(z):
    return np.arctan(z)

def arctan_derivative(z):
    return 1.0/(np.square(z)+1.0)