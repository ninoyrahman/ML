'''
Created on Feb 06, 2025

@author: ninoy
'''
import numpy as np
import cupy as cp
from neural_network_cupy import NN as NN
from neural_network import NN as cNN

class adam(NN):
    def __init__(self, label_number, 
                 alpha=0.1, 
                 epoch=500, 
                 activation='ReLU', 
                 layer_size=[784, 10, 10, 10], 
                 accuracy=0.9, 
                 batch_size=32, 
                 gradient_clip=None, 
                 beta1=0.9, 
                 beta2=0.999):
        self.__label_number = label_number # number of labels
        self.__alpha = alpha # learning rate
        self.__epoch = epoch # epoch number
        self.__activation = activation # activation function
        self.__layer_size = layer_size # number of hidden layer
        self.__accuracy = accuracy # accuracy required
        self.__batch_size = batch_size # batch size for SGD
        self.__gradient_clip = gradient_clip # clip value for gradient
        self.__eps = 1e-8 # parameter for RMSprop and Adam
        self.__beta1 = beta1 # parameter for Adam
        self.__beta2 = beta2 # parameter for Adam
        super().__init__(label_number, alpha=alpha, epoch=epoch, activation=activation, layer_size=layer_size, accuracy=accuracy, batch_size=batch_size, gradient_clip=gradient_clip, beta1=beta1, beta2=beta2)

    # conduct adam 
    def train(self, X, Y):
        
        w1, b1, w2, b2, w3, b3 = self.initialze_parameters()
        
        v_w1, v_b1, v_w2, v_b2, v_w3, v_b3 = 0., 0., 0., 0., 0., 0.
        
        m_w1, m_b1, m_w2, m_b2, m_w3, m_b3 = 0., 0., 0., 0., 0., 0.
        
        vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3 = 0., 0., 0., 0., 0., 0.
        
        mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3 = 0., 0., 0., 0., 0., 0.
        
        for i in range(self.__epoch):

            # shuffle data
            data = cp.c_[Y, X.T]
            cp.random.shuffle(data)
            data = data.T
            Y_new = cp.array(data[0, :], dtype=cp.int32)
            X_new = data[1:, :]

            for j in range(0, Y.size, self.__batch_size):

                # select batch
                X_batch = X_new[:, j:j+self.__batch_size]
                Y_batch = Y_new[j:j+self.__batch_size]
            
                z1, a1, z2, a2, z3, a3 = self.__forward_propagation__(w1, b1, w2, b2, w3, b3, X_batch)
                dw1, db1, dw2, db2, dw3, db3 = self.__backward_propagation__(z1, a1, z2, a2, z3, a3, w1, w2, w3, X_batch, Y_batch)
                m_w1, m_b1, m_w2, m_b2, m_w3, m_b3, v_w1, v_b1, v_w2, v_b2, v_w3, v_b3, mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3, vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3, w1, b1, w2, b2, w3, b3 = self.__update_parameters_Adam__(i+1, m_w1, m_b1, m_w2, m_b2, m_w3, m_b3, v_w1, v_b1, v_w2, v_b2, v_w3, v_b3, mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3, vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3, w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3)
            
            
            predictions = self.predictions(X, w1, b1, w2, b2, w3, b3)
            acc = self.get_accuracy(predictions, Y)
            if i % 100 == 0:
                print("Epoch: ", i, "Accuracy: ", acc)
            if acc > self.__accuracy:
                print("Epoch: ", i, "Accuracy: ", acc)
                return w1, b1, w2, b2, w3, b3

        return w1, b1, w2, b2, w3, b3

class adam_cpu(cNN):
    def __init__(self, label_number, 
                 alpha=0.1, 
                 epoch=500, 
                 activation='ReLU', 
                 layer_size=[784, 10, 10, 10], 
                 accuracy=0.9, 
                 batch_size=32, 
                 gradient_clip=None, 
                 beta1=0.9, 
                 beta2=0.999):
        self.__label_number = label_number # number of labels
        self.__alpha = alpha # learning rate
        self.__epoch = epoch # epoch number
        self.__activation = activation # activation function
        self.__layer_size = layer_size # number of hidden layer
        self.__accuracy = accuracy # accuracy required
        self.__batch_size = batch_size # batch size for SGD
        self.__gradient_clip = gradient_clip # clip value for gradient
        self.__eps = 1e-8 # parameter for RMSprop and Adam
        self.__beta1 = beta1 # parameter for Adam
        self.__beta2 = beta2 # parameter for Adam
        super().__init__(label_number, alpha=alpha, epoch=epoch, activation=activation, layer_size=layer_size, accuracy=accuracy, batch_size=batch_size, gradient_clip=gradient_clip, beta1=beta1, beta2=beta2)

    # conduct adam 
    def train(self, X, Y):
        
        w1, b1, w2, b2, w3, b3 = self.initialze_parameters()
        
        v_w1, v_b1, v_w2, v_b2, v_w3, v_b3 = 0., 0., 0., 0., 0., 0.
        
        m_w1, m_b1, m_w2, m_b2, m_w3, m_b3 = 0., 0., 0., 0., 0., 0.
        
        vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3 = 0., 0., 0., 0., 0., 0.
        
        mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3 = 0., 0., 0., 0., 0., 0.
        
        for i in range(self.__epoch):

            # shuffle data
            data = np.c_[Y, X.T]
            np.random.shuffle(data)
            data = data.T
            Y_new = np.array(data[0, :], dtype=np.int32)
            X_new = data[1:, :]

            for j in range(0, Y.size, self.__batch_size):

                # select batch
                X_batch = X_new[:, j:j+self.__batch_size]
                Y_batch = Y_new[j:j+self.__batch_size]
            
                z1, a1, z2, a2, z3, a3 = self.__forward_propagation__(w1, b1, w2, b2, w3, b3, X_batch)
                dw1, db1, dw2, db2, dw3, db3 = self.__backward_propagation__(z1, a1, z2, a2, z3, a3, w1, w2, w3, X_batch, Y_batch)
                m_w1, m_b1, m_w2, m_b2, m_w3, m_b3, v_w1, v_b1, v_w2, v_b2, v_w3, v_b3, mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3, vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3, w1, b1, w2, b2, w3, b3 = self.__update_parameters_Adam__(i+1, m_w1, m_b1, m_w2, m_b2, m_w3, m_b3, v_w1, v_b1, v_w2, v_b2, v_w3, v_b3, mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3, vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3, w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3)
            
            
            predictions = self.predictions(X, w1, b1, w2, b2, w3, b3)
            acc = self.get_accuracy(predictions, Y)
            if i % 100 == 0:
                print("Epoch: ", i, "Accuracy: ", acc)
            if acc > self.__accuracy:
                print("Epoch: ", i, "Accuracy: ", acc)
                return w1, b1, w2, b2, w3, b3

        return w1, b1, w2, b2, w3, b3        
