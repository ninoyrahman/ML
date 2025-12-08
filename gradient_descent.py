'''
Created on Feb 06, 2025

@author: ninoy
'''
import numpy as np
import cupy as cp
from neural_network_cupy import NN as NN
from neural_network import NN as cNN

class gradient_descent(NN):
    def __init__(self, label_number, 
                 alpha=0.1, 
                 epoch=500, 
                 activation='ReLU', 
                 layer_size=[784, 10, 10, 10], 
                 accuracy=0.9, 
                 gradient_clip=None,
                 beta=0.9):
        self.__label_number = label_number # number of labels
        self.__alpha = alpha # learning rate
        self.__epoch = epoch # epoch number
        self.__activation = activation # activation function
        self.__layer_size = layer_size # number of hidden layer
        self.__accuracy = accuracy # accuracy required
        self.__gradient_clip = gradient_clip # clip value for gradient
        self.__beta = beta # parameter for RMSprop 
        super().__init__(label_number, alpha=alpha, epoch=epoch, activation=activation, layer_size=layer_size, accuracy=accuracy, gradient_clip=gradient_clip, beta=beta)

    # conduct gradient descent
    def train(self, X, Y):
        w1, b1, w2, b2, w3, b3 = self.initialze_parameters()
        
        for i in range(self.__epoch):
            z1, a1, z2, a2, z3, a3 = self.__forward_propagation__(w1, b1, w2, b2, w3, b3, X)
            dw1, db1, dw2, db2, dw3, db3 = self.__backward_propagation__(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y)
            w1, b1, w2, b2, w3, b3 = self.__update_parameters__(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3)
            
            predictions = self.predictions(X, w1, b1, w2, b2, w3, b3)
            acc = self.get_accuracy(predictions, Y)
            if i % 100 == 0:
                print("Epoch: ", i, "Accuracy: ", acc)
            if acc > self.__accuracy:
                print("Epoch: ", i, "Accuracy: ", acc)
                return w1, b1, w2, b2, w3, b3
                
        return w1, b1, w2, b2, w3, b3
    
class gradient_descent_cpu(cNN):
    def __init__(self, label_number, 
                 alpha=0.1, 
                 epoch=500, 
                 activation='ReLU', 
                 layer_size=[784, 10, 10, 10], 
                 accuracy=0.9, 
                 gradient_clip=None,
                 beta=0.9):
        self.__label_number = label_number # number of labels
        self.__alpha = alpha # learning rate
        self.__epoch = epoch # epoch number
        self.__activation = activation # activation function
        self.__layer_size = layer_size # number of hidden layer
        self.__accuracy = accuracy # accuracy required
        self.__gradient_clip = gradient_clip # clip value for gradient
        self.__beta = beta # parameter for RMSprop 
        super().__init__(label_number, alpha=alpha, epoch=epoch, activation=activation, layer_size=layer_size, accuracy=accuracy, gradient_clip=gradient_clip, beta=beta)

    # conduct gradient descent
    def train(self, X, Y):
        w1, b1, w2, b2, w3, b3 = self.initialze_parameters()
        
        for i in range(self.__epoch):
            z1, a1, z2, a2, z3, a3 = self.__forward_propagation__(w1, b1, w2, b2, w3, b3, X)
            dw1, db1, dw2, db2, dw3, db3 = self.__backward_propagation__(z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y)
            w1, b1, w2, b2, w3, b3 = self.__update_parameters__(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3)
            
            predictions = self.predictions(X, w1, b1, w2, b2, w3, b3)
            acc = self.get_accuracy(predictions, Y)
            if i % 100 == 0:
                print("Epoch: ", i, "Accuracy: ", acc)
            if acc > self.__accuracy:
                print("Epoch: ", i, "Accuracy: ", acc)
                return w1, b1, w2, b2, w3, b3
                
        return w1, b1, w2, b2, w3, b3