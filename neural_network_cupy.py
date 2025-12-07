'''
Created on Feb 06, 2025

@author: ninoy
'''
import numpy as np
import cupy as cp

# class for neural network with two hidden layer
class NN:
    def __init__(self, label_number, 
                 alpha=0.1, 
                 epoch=500, 
                 activation='ReLU', 
                 layer_size=[784, 10, 10, 10], 
                 accuracy=0.9, 
                 batch_size=32, 
                 gradient_clip=None, 
                 gamma=0.9, 
                 beta=0.9, 
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
        self.__gamma = gamma # parameter for SGD with momentum
        self.__beta = beta # parameter for RMSprop 
        self.__eps = 1e-8 # parameter for RMSprop and Adam
        self.__beta1 = beta1 # parameter for Adam
        self.__beta2 = beta2 # parameter for Adam

    # weight and biases initialization
    def initialze_parameters(self):
        cp.random.seed(42)
        w1 = cp.random.normal(size=(self.__layer_size[1], self.__layer_size[0])).astype(cp.float64) * cp.sqrt(1. / self.__layer_size[0])
        b1 = cp.random.normal(size=(self.__layer_size[1], 1)).astype(cp.float64) * cp.sqrt(1. / self.__layer_size[1])
        w2 = cp.random.normal(size=(self.__layer_size[2], self.__layer_size[1])).astype(cp.float64) * cp.sqrt(1. / ( self.__layer_size[2] * 2. ))
        b2 = cp.random.normal(size=(self.__layer_size[2], 1)).astype(cp.float64) * cp.sqrt(1. / self.__layer_size[2])
        w3 = cp.random.normal(size=(self.__layer_size[3], self.__layer_size[2])).astype(cp.float64) * cp.sqrt(1. / ( self.__layer_size[3] * 2. ))
        b3 = cp.random.normal(size=(self.__layer_size[3], 1)).astype(cp.float64) * cp.sqrt(1. / self.__layer_size[3])
        return w1, b1, w2, b2, w3, b3

    # print model parameters
    def print_parameter(self):
        print('')
        print('NN parameters:')
        print('number of labels   = ', self.__label_number)
        print('epoch              = ', self.__epoch)
        print('learning_rate      = ', self.__alpha)
        print('activation         = ', self.__activation)
        print('accuracy           = ', self.__accuracy)
        print('batch size for SGD = ', self.__batch_size)
        print('gradient clip      = ', self.__gradient_clip)
        print('gamma              = ', self.__gamma)
        print('beta               = ', self.__beta)
        print('beta1              = ', self.__beta1)
        print('beta2              = ', self.__beta2)
        print('')        

    # activation function ReLU/sigmoid
    def factivation(self, z):
        if self.__activation == 'ReLU':
            return cp.maximum(z, 0)
        else:
            return 1.0 / (1.0 + cp.exp(-z))

    # derivative of activation function ReLU/sigmoid
    def dfactivation(self, z):
        if self.__activation == 'ReLU':
            return z > 0
        else:
            return self.factivation(z) * (1.0 - self.factivation(z))

    # softmax function at output layer
    def softmax(self, z):
        return cp.exp(z) / sum(cp.exp(z))

    # softmax function at output layer
    def dsoftmax(self, z):
        return self.softmax(z) * ( 1.0 - self.softmax(z) )

    # forward propagation
    def __forward_propagation__(self, w1, b1, w2, b2, w3, b3, X):

        # hidden layer 1
        z1 = w1.dot(X) + b1  # (N1, N0) x (N0, Ns) + (N1) = (N1, Ns)
        a1 = self.factivation(z1)

        # hidden layer 2
        z2 = w2.dot(a1) + b2 # (N2, N1) x (N1, Ns) + (N2) = (N2, Ns)
        a2 = self.factivation(z2)

        # output layer
        z3 = w3.dot(a2) + b3 # (N3, N2) x (N2, Ns) + (N3) = (N3, Ns)
        a3 = self.softmax(z3)
        
        return z1, a1, z2, a2, z3, a3

    # label to index transform
    def __one_hot__(self, Y):
        one_hot_Y = cp.zeros((Y.size, self.__label_number))
        one_hot_Y[cp.arange(Y.size), Y] = 1
        # one_hot_Y[:, 0] = Y
        return one_hot_Y.T

    # clip gradient
    def __gradient_clipping__(self, dw1, db1, dw2, db2, dw3, db3):
        dw1 = cp.clip(dw1, -self.__gradient_clip, self.__gradient_clip)
        db1 = cp.clip(db1, -self.__gradient_clip, self.__gradient_clip)
        dw2 = cp.clip(dw2, -self.__gradient_clip, self.__gradient_clip)
        db2 = cp.clip(db2, -self.__gradient_clip, self.__gradient_clip)
        dw3 = cp.clip(dw3, -self.__gradient_clip, self.__gradient_clip)
        db3 = cp.clip(db3, -self.__gradient_clip, self.__gradient_clip)
        return dw1, db1, dw2, db2, dw3, db3
        
    # backward propagation
    def __backward_propagation__(self, z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y):
        m = Y.size
        one_hot_Y = self.__one_hot__(Y)

        # output layer to hidden layer 2
        delta = (1.0 / m) * (a3 - one_hot_Y)
        dw3 = delta.dot(a2.T)
        db3 = cp.array([cp.sum(delta)])

        # hidden layer 2 to hidden layer 1
        delta1 = w3.T.dot(delta) * self.dfactivation(z2)
        dw2 = delta1.dot(a1.T)
        db2 = cp.array([cp.sum(delta1)])

        # hidden layer 1 to input layer
        delta2 = w2.T.dot(delta1) * self.dfactivation(z1)
        dw1 = delta2.dot(X.T)
        db1 = cp.array([cp.sum(delta2)])

        if self.__gradient_clip != None:
            return self.__gradient_clipping__(dw1, db1, dw2, db2, dw3, db3)
        
        return dw1, db1, dw2, db2, dw3, db3

    # weights and biases update
    def __update_parameters__(self, w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3):
        w1 = w1 - self.__alpha * dw1
        b1 = b1 - self.__alpha * db1
        w2 = w2 - self.__alpha * dw2
        b2 = b2 - self.__alpha * db2
        w3 = w3 - self.__alpha * dw3
        b3 = b3 - self.__alpha * db3        
        return w1, b1, w2, b2, w3, b3

    # velocities, weights and biases update for Adam
    def __update_parameters_Adam__(self, t, m_w1, m_b1, m_w2, m_b2, m_w3, m_b3,
                                            v_w1, v_b1, v_w2, v_b2, v_w3, v_b3,
                                            mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3,
                                            vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3,
                                            w1, b1, w2, b2, w3, b3,
                                            dw1, db1, dw2, db2, dw3, db3):

        m_w1 = self.__beta1 * m_w1 + (1. - self.__beta1) * dw1
        m_b1 = self.__beta1 * m_b1 + (1. - self.__beta1) * db1
        m_w2 = self.__beta1 * m_w2 + (1. - self.__beta1) * dw2
        m_b2 = self.__beta1 * m_b2 + (1. - self.__beta1) * db2
        m_w3 = self.__beta1 * m_w3 + (1. - self.__beta1) * dw3
        m_b3 = self.__beta1 * m_b3 + (1. - self.__beta1) * db3
        
        v_w1 = self.__beta2 * v_w1 + (1. - self.__beta2) * dw1**2
        v_b1 = self.__beta2 * v_b1 + (1. - self.__beta2) * db1**2
        v_w2 = self.__beta2 * v_w2 + (1. - self.__beta2) * dw2**2
        v_b2 = self.__beta2 * v_b2 + (1. - self.__beta2) * db2**2
        v_w3 = self.__beta2 * v_w3 + (1. - self.__beta2) * dw3**2
        v_b3 = self.__beta2 * v_b3 + (1. - self.__beta2) * db3**2

        mhat_w1 = m_w1 / (1. - self.__beta1**t)
        mhat_b1 = m_b1 / (1. - self.__beta1**t)
        mhat_w2 = m_w2 / (1. - self.__beta1**t)
        mhat_b2 = m_b2 / (1. - self.__beta1**t)
        mhat_w3 = m_w3 / (1. - self.__beta1**t)
        mhat_b3 = m_b3 / (1. - self.__beta1**t)

        vhat_w1 = v_w1 / (1. - self.__beta2**t)
        vhat_b1 = v_b1 / (1. - self.__beta2**t)
        vhat_w2 = v_w2 / (1. - self.__beta2**t)
        vhat_b2 = v_b2 / (1. - self.__beta2**t)
        vhat_w3 = v_w3 / (1. - self.__beta2**t)
        vhat_b3 = v_b3 / (1. - self.__beta2**t)
        
        w1 = w1 - self.__alpha * mhat_w1 / (cp.sqrt(vhat_w1) + self.__eps)
        b1 = b1 - self.__alpha * mhat_b1 / (cp.sqrt(vhat_b1) + self.__eps)
        w2 = w2 - self.__alpha * mhat_w2 / (cp.sqrt(vhat_w2) + self.__eps)
        b2 = b2 - self.__alpha * mhat_b2 / (cp.sqrt(vhat_b2) + self.__eps)
        w3 = w3 - self.__alpha * mhat_w3 / (cp.sqrt(vhat_w3) + self.__eps)
        b3 = b3 - self.__alpha * mhat_b3 / (cp.sqrt(vhat_b1) + self.__eps)
        
        return m_w1, m_b1, m_w2, m_b2, m_w3, m_b3, v_w1, v_b1, v_w2, v_b2, v_w3, v_b3, mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3, vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3, w1, b1, w2, b2, w3, b3

    # get prediction
    def __get_predictions__(self, a3):
        return cp.argmax(a3, 0)

    # get accuracy
    def get_accuracy(self, predictions, Y):
        return cp.sum(predictions == Y) / Y.size
        # return cp.sum(cp.abs(predictions/Y - 1.)) / Y.size

    # get loss
    def get_loss(self, predictions, Y):
        return ( 0.5 * (predictions - Y)**2 ).sum() / Y.size

    # get ce loss
    def get_ce_loss(self, a, Y):
        return -cp.sum( cp.log( a[Y, range(a.shape[1])] ) ) / Y.size

    # evaluate prediction
    def predictions(self, X, w1, b1, w2, b2, w3, b3):
        _, _, _, _, _, a3 = self.__forward_propagation__(w1, b1, w2, b2, w3, b3, X)
        predictions = self.__get_predictions__(a3)
        # predictions = a3[0, :]
        return predictions