'''
Created on Feb 06, 2025

@author: ninoy
'''
import numpy as np
import cupy as cp

class NN:
    """
    class for neural network with two hidden layer

    ...

    Attributes
    ----------

    Methods
    -------
    initialze_parameters(self):
        weight and biases initialization
    print_parameter(self):
        print model parameters
    factivation(self, z)
        activation function ReLU/sigmoid
    dfactivation(self, z)
        derivative of activation function ReLU/sigmoid
    softmax(self, z)
        softmax function at output layer
    dsoftmax(self, z)
        softmax function at output layer
    __forward_propagation__(self, w1, b1, w2, b2, w3, b3, X)
        forward propagation
    __one_hot__(self, Y)
        label to index transform
    __gradient_clipping__(self, dw1, db1, dw2, db2, dw3, db3)
        clip gradient
    __backward_propagation__(self, z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y)
        backward propagation
    __update_parameters__(self, w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3)
        weights and biases update
    __update_parameters_Adam__(self, t, m_w1, m_b1, m_w2, m_b2, m_w3, m_b3,
                                            v_w1, v_b1, v_w2, v_b2, v_w3, v_b3,
                                            mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3,
                                            vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3,
                                            w1, b1, w2, b2, w3, b3,
                                            dw1, db1, dw2, db2, dw3, db3)
        velocities, weights and biases update for Adam
    __get_predictions__(self, a3)
        get prediction
    get_accuracy(self, predictions, Y)
        get accuracy
    get_loss(self, predictions, Y)
        get loss
    get_ce_loss(self, a, Y)
        get ce loss
    predictions(self, X, w1, b1, w2, b2, w3, b3)
        evaluate prediction
    """
    def __init__(self, label_number, 
                 alpha=0.1, 
                 epoch=500, 
                 activation='ReLU', 
                 layer_size=[784, 10, 10, 10], 
                 accuracy=0.9, 
                 batch_size=32, 
                 gradient_clip=None,
                 beta1=0.9, 
                 beta2=0.999,
                 gpu=True):
        """
        Parameters
        ----------
        label_number : int
            number of labels
        alpha : float
            learning rate
        epoch : int
            epoch number
        activation : str
            activation function
        layer_size : list
            number of hidden layer
        accuracy : float
            accuracy required
        batch_size : int
            batch size for SGD
        gradient_clip : float
            clip value for gradient
        beta1 : float
            parameter for Adam
        beta2 : float
            parameter for Adam
        gpu : bool
            flag for gpu usage
        """
        self.label_number = label_number # number of labels
        self.alpha = alpha # learning rate
        self.epoch = epoch # epoch number
        self.activation = activation # activation function
        self.layer_size = layer_size # number of hidden layer
        self.accuracy = accuracy # accuracy required
        self.batch_size = batch_size # batch size for SGD
        self.gradient_clip = gradient_clip # clip value for gradient
        self.eps = 1e-8 # parameter for Adam
        self.beta1 = beta1 # parameter for Adam
        self.beta2 = beta2 # parameter for Adam
        self.cls = np # cp for gpu, np for cpu
        if gpu:
            self.cls = cp

    # weight and biases initialization
    def initialze_parameters(self):
        """
        Parameters
        ----------
        """
        self.cls.random.seed(42)
        w1 = self.cls.random.normal(size=(self.layer_size[1], self.layer_size[0])).astype(self.cls.float64) * self.cls.sqrt(1. / self.layer_size[0])
        b1 = self.cls.random.normal(size=(self.layer_size[1], 1)).astype(self.cls.float64) * self.cls.sqrt(1. / self.layer_size[1])
        w2 = self.cls.random.normal(size=(self.layer_size[2], self.layer_size[1])).astype(self.cls.float64) * self.cls.sqrt(1. / ( self.layer_size[2] * 2. ))
        b2 = self.cls.random.normal(size=(self.layer_size[2], 1)).astype(self.cls.float64) * self.cls.sqrt(1. / self.layer_size[2])
        w3 = self.cls.random.normal(size=(self.layer_size[3], self.layer_size[2])).astype(self.cls.float64) * self.cls.sqrt(1. / ( self.layer_size[3] * 2. ))
        b3 = self.cls.random.normal(size=(self.layer_size[3], 1)).astype(self.cls.float64) * self.cls.sqrt(1. / self.layer_size[3])
        return w1, b1, w2, b2, w3, b3

    # print model parameters
    def print_parameter(self):
        """
        Parameters
        ----------
        """
        print('')
        print('NN parameters:')
        print('number of labels   = ', self.label_number)
        print('epoch              = ', self.epoch)
        print('learning_rate      = ', self.alpha)
        print('activation         = ', self.activation)
        print('accuracy           = ', self.accuracy)
        print('batch size for SGD = ', self.batch_size)
        print('gradient clip      = ', self.gradient_clip)
        print('beta1              = ', self.beta1)
        print('beta2              = ', self.beta2)
        print('')        

    # activation function ReLU/sigmoid
    def factivation(self, z):
        """
        Parameters
        ----------
        z : numpy.ndarray or cupy.ndarray
            argument of activation function
        """
        if self.activation == 'ReLU':
            return self.cls.maximum(z, 0)
        else:
            return 1.0 / (1.0 + self.cls.exp(-z))

    # derivative of activation function ReLU/sigmoid
    def dfactivation(self, z):
        """
        Parameters
        ----------
        z : numpy.ndarray or cupy.ndarray
            argument of activation function
        """
        if self.activation == 'ReLU':
            return z > 0
        else:
            return self.factivation(z) * (1.0 - self.factivation(z))

    # softmax function at output layer
    def softmax(self, z):
        """
        Parameters
        ----------
        z : numpy.ndarray or cupy.ndarray
            argument of softmax function
        """
        return self.cls.exp(z) / sum(self.cls.exp(z))

    # softmax function at output layer
    def dsoftmax(self, z):
        """
        Parameters
        ----------
        z : numpy.ndarray or cupy.ndarray
            argument of softmax function
        """
        return self.softmax(z) * ( 1.0 - self.softmax(z) )

    # forward propagation
    def __forward_propagation__(self, w1, b1, w2, b2, w3, b3, X):
        """
        Parameters
        ----------
        w1 : numpy.ndarray or cupy.ndarray
            weights
        b1 : numpy.ndarray or cupy.ndarray
            biases
        w2 : numpy.ndarray or cupy.ndarray
            weights
        b2 : numpy.ndarray or cupy.ndarray
            biases
        w3 : numpy.ndarray or cupy.ndarray
            weights
        b3 : numpy.ndarray or cupy.ndarray
            biases
        X : numpy.ndarray or cupy.ndarray
            features
        Returns
        -------
        z1 : numpy.ndarray or cupy.ndarray
            intermediate variable
        a1 : numpy.ndarray or cupy.ndarray
            hidden activation vector
        z2 : numpy.ndarray or cupy.ndarray
            intermediate variable
        a2 : numpy.ndarray or cupy.ndarray
            hidden activation vector
        z3 : numpy.ndarray or cupy.ndarray
            intermediate variable
        a3 : numpy.ndarray or cupy.ndarray
            output vector
        """
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
        """
        Parameters
        ----------
        Y : numpy.ndarray or cupy.ndarray
            labels
        Returns
        -------
        one_hot_Y : numpy.ndarray or cupy.ndarray
            one hot of labels
        """
        one_hot_Y = self.cls.zeros((Y.size, self.label_number))
        one_hot_Y[self.cls.arange(Y.size), Y] = 1
        return one_hot_Y.T

    # clip gradient
    def __gradient_clipping__(self, dw1, db1, dw2, db2, dw3, db3):
        """
        Parameters
        ----------
        dw1 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db1 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw2 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db2 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw3 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db3 : numpy.ndarray or cupy.ndarray
            gradient of biases
        Returns
        -------
        dw1 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db1 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw2 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db2 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw3 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db3 : numpy.ndarray or cupy.ndarray
            gradient of biases
        """
        dw1 = self.cls.clip(dw1, -self.gradient_clip, self.gradient_clip)
        db1 = self.cls.clip(db1, -self.gradient_clip, self.gradient_clip)
        dw2 = self.cls.clip(dw2, -self.gradient_clip, self.gradient_clip)
        db2 = self.cls.clip(db2, -self.gradient_clip, self.gradient_clip)
        dw3 = self.cls.clip(dw3, -self.gradient_clip, self.gradient_clip)
        db3 = self.cls.clip(db3, -self.gradient_clip, self.gradient_clip)
        return dw1, db1, dw2, db2, dw3, db3
        
    # backward propagation
    def __backward_propagation__(self, z1, a1, z2, a2, z3, a3, w1, w2, w3, X, Y):
        """
        Parameters
        ----------
        z1 : numpy.ndarray or cupy.ndarray
            intermediate variable
        a1 : numpy.ndarray or cupy.ndarray
            hidden activation vector
        z2 : numpy.ndarray or cupy.ndarray
            intermediate variable
        a2 : numpy.ndarray or cupy.ndarray
            hidden activation vector
        z3 : numpy.ndarray or cupy.ndarray
            intermediate variable
        a3 : numpy.ndarray or cupy.ndarray
            output vector
        w1 : numpy.ndarray or cupy.ndarray
            weights
        w2 : numpy.ndarray or cupy.ndarray
            weights
        w3 : numpy.ndarray or cupy.ndarray
            weights
        X : numpy.ndarray or cupy.ndarray
            features
        Y : numpy.ndarray or cupy.ndarray
            labels
        Returns
        -------
        dw1 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db1 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw2 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db2 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw3 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db3 : numpy.ndarray or cupy.ndarray
            gradient of biases
        """
        m = Y.size
        one_hot_Y = self.__one_hot__(Y)

        # output layer to hidden layer 2
        delta = (1.0 / m) * (a3 - one_hot_Y)
        dw3 = delta.dot(a2.T)
        db3 = self.cls.array([self.cls.sum(delta)])

        # hidden layer 2 to hidden layer 1
        delta1 = w3.T.dot(delta) * self.dfactivation(z2)
        dw2 = delta1.dot(a1.T)
        db2 = self.cls.array([self.cls.sum(delta1)])

        # hidden layer 1 to input layer
        delta2 = w2.T.dot(delta1) * self.dfactivation(z1)
        dw1 = delta2.dot(X.T)
        db1 = self.cls.array([self.cls.sum(delta2)])

        if self.gradient_clip != None:
            return self.__gradient_clipping__(dw1, db1, dw2, db2, dw3, db3)
        
        return dw1, db1, dw2, db2, dw3, db3

    # weights and biases update
    def __update_parameters__(self, w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3):
        """
        Parameters
        ----------
        w1 : numpy.ndarray or cupy.ndarray
            weights
        b1 : numpy.ndarray or cupy.ndarray
            biases
        w2 : numpy.ndarray or cupy.ndarray
            weights
        b2 : numpy.ndarray or cupy.ndarray
            biases
        w3 : numpy.ndarray or cupy.ndarray
            weights
        b3 : numpy.ndarray or cupy.ndarray
            biases
        dw1 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db1 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw2 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db2 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw3 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db3 : numpy.ndarray or cupy.ndarray
            gradient of biases
        Returns
        -------
        w1 : numpy.ndarray or cupy.ndarray
            weights
        b1 : numpy.ndarray or cupy.ndarray
            biases
        w2 : numpy.ndarray or cupy.ndarray
            weights
        b2 : numpy.ndarray or cupy.ndarray
            biases
        w3 : numpy.ndarray or cupy.ndarray
            weights
        b3 : numpy.ndarray or cupy.ndarray
            biases
        """
        w1 = w1 - self.alpha * dw1
        b1 = b1 - self.alpha * db1
        w2 = w2 - self.alpha * dw2
        b2 = b2 - self.alpha * db2
        w3 = w3 - self.alpha * dw3
        b3 = b3 - self.alpha * db3        
        return w1, b1, w2, b2, w3, b3

    # velocities, weights and biases update for Adam
    def __update_parameters_Adam__(self, t, m_w1, m_b1, m_w2, m_b2, m_w3, m_b3,
                                            v_w1, v_b1, v_w2, v_b2, v_w3, v_b3,
                                            mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3,
                                            vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3,
                                            w1, b1, w2, b2, w3, b3,
                                            dw1, db1, dw2, db2, dw3, db3):
        """
        Parameters
        ----------
        t : int
            epoch
        m_w1 : numpy.ndarray or cupy.ndarray
            weight momentum
        m_b1 : numpy.ndarray or cupy.ndarray
            bias momentum
        m_w2 : numpy.ndarray or cupy.ndarray
            weight momentum
        m_b2 : numpy.ndarray or cupy.ndarray
            bias momentum
        m_w3 : numpy.ndarray or cupy.ndarray
            weight momentum
        m_b3 : numpy.ndarray or cupy.ndarray
            bias momentum
        v_w1 : numpy.ndarray or cupy.ndarray
            weight velocity
        v_b1 : numpy.ndarray or cupy.ndarray
            bias velocity
        v_w2 : numpy.ndarray or cupy.ndarray
            weight velocity
        v_b2 : numpy.ndarray or cupy.ndarray
            bias velocity
        v_w3 : numpy.ndarray or cupy.ndarray
            weight velocity
        v_b3 : numpy.ndarray or cupy.ndarray
            bias velocity
        mhat_w1 : numpy.ndarray or cupy.ndarray
            bias-corrected weight momentum
        mhat_b1 : numpy.ndarray or cupy.ndarray
            bias-corrected bias momentum
        mhat_w2 : numpy.ndarray or cupy.ndarray
            bias-corrected weight momentum
        mhat_b2 : numpy.ndarray or cupy.ndarray
            bias-corrected bias momentum
        mhat_w3 : numpy.ndarray or cupy.ndarray
            bias-corrected weight momentum
        mhat_b3 : numpy.ndarray or cupy.ndarray
            bias-corrected bias momentum
        vhat_w1 : numpy.ndarray or cupy.ndarray
            bias-corrected weight velocity
        vhat_b1 : numpy.ndarray or cupy.ndarray
            bias-corrected bias velocity
        vhat_w2 : numpy.ndarray or cupy.ndarray
            bias-corrected weight velocity
        vhat_b2 : numpy.ndarray or cupy.ndarray
            bias-corrected bias velocity
        vhat_w3 : numpy.ndarray or cupy.ndarray
            bias-corrected weight velocity
        vhat_b3 : numpy.ndarray or cupy.ndarray
            bias-corrected bias velocity
        w1 : numpy.ndarray or cupy.ndarray
            weights
        b1 : numpy.ndarray or cupy.ndarray
            biases
        w2 : numpy.ndarray or cupy.ndarray
            weights
        b2 : numpy.ndarray or cupy.ndarray
            biases
        w3 : numpy.ndarray or cupy.ndarray
            weights
        b3 : numpy.ndarray or cupy.ndarray
            biases
        dw1 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db1 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw2 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db2 : numpy.ndarray or cupy.ndarray
            gradient of biases
        dw3 : numpy.ndarray or cupy.ndarray
            gradient of weights
        db3 : numpy.ndarray or cupy.ndarray
            gradient of biases
        Returns
        -------
        m_w1 : numpy.ndarray or cupy.ndarray
            weight momentum
        m_b1 : numpy.ndarray or cupy.ndarray
            bias momentum
        m_w2 : numpy.ndarray or cupy.ndarray
            weight momentum
        m_b2 : numpy.ndarray or cupy.ndarray
            bias momentum
        m_w3 : numpy.ndarray or cupy.ndarray
            weight momentum
        m_b3 : numpy.ndarray or cupy.ndarray
            bias momentum
        v_w1 : numpy.ndarray or cupy.ndarray
            weight velocity
        v_b1 : numpy.ndarray or cupy.ndarray
            bias velocity
        v_w2 : numpy.ndarray or cupy.ndarray
            weight velocity
        v_b2 : numpy.ndarray or cupy.ndarray
            bias velocity
        v_w3 : numpy.ndarray or cupy.ndarray
            weight velocity
        v_b3 : numpy.ndarray or cupy.ndarray
            bias velocity
        mhat_w1 : numpy.ndarray or cupy.ndarray
            bias-corrected weight momentum
        mhat_b1 : numpy.ndarray or cupy.ndarray
            bias-corrected bias momentum
        mhat_w2 : numpy.ndarray or cupy.ndarray
            bias-corrected weight momentum
        mhat_b2 : numpy.ndarray or cupy.ndarray
            bias-corrected bias momentum
        mhat_w3 : numpy.ndarray or cupy.ndarray
            bias-corrected weight momentum
        mhat_b3 : numpy.ndarray or cupy.ndarray
            bias-corrected bias momentum
        vhat_w1 : numpy.ndarray or cupy.ndarray
            bias-corrected weight velocity
        vhat_b1 : numpy.ndarray or cupy.ndarray
            bias-corrected bias velocity
        vhat_w2 : numpy.ndarray or cupy.ndarray
            bias-corrected weight velocity
        vhat_b2 : numpy.ndarray or cupy.ndarray
            bias-corrected bias velocity
        vhat_w3 : numpy.ndarray or cupy.ndarray
            bias-corrected weight velocity
        vhat_b3 : numpy.ndarray or cupy.ndarray
            bias-corrected bias velocity
        w1 : numpy.ndarray or cupy.ndarray
            weights
        b1 : numpy.ndarray or cupy.ndarray
            biases
        w2 : numpy.ndarray or cupy.ndarray
            weights
        b2 : numpy.ndarray or cupy.ndarray
            biases
        w3 : numpy.ndarray or cupy.ndarray
            weights
        b3 : numpy.ndarray or cupy.ndarray
            biases
        """
        m_w1 = self.beta1 * m_w1 + (1. - self.beta1) * dw1
        m_b1 = self.beta1 * m_b1 + (1. - self.beta1) * db1
        m_w2 = self.beta1 * m_w2 + (1. - self.beta1) * dw2
        m_b2 = self.beta1 * m_b2 + (1. - self.beta1) * db2
        m_w3 = self.beta1 * m_w3 + (1. - self.beta1) * dw3
        m_b3 = self.beta1 * m_b3 + (1. - self.beta1) * db3
        
        v_w1 = self.beta2 * v_w1 + (1. - self.beta2) * dw1**2
        v_b1 = self.beta2 * v_b1 + (1. - self.beta2) * db1**2
        v_w2 = self.beta2 * v_w2 + (1. - self.beta2) * dw2**2
        v_b2 = self.beta2 * v_b2 + (1. - self.beta2) * db2**2
        v_w3 = self.beta2 * v_w3 + (1. - self.beta2) * dw3**2
        v_b3 = self.beta2 * v_b3 + (1. - self.beta2) * db3**2

        mhat_w1 = m_w1 / (1. - self.beta1**t)
        mhat_b1 = m_b1 / (1. - self.beta1**t)
        mhat_w2 = m_w2 / (1. - self.beta1**t)
        mhat_b2 = m_b2 / (1. - self.beta1**t)
        mhat_w3 = m_w3 / (1. - self.beta1**t)
        mhat_b3 = m_b3 / (1. - self.beta1**t)

        vhat_w1 = v_w1 / (1. - self.beta2**t)
        vhat_b1 = v_b1 / (1. - self.beta2**t)
        vhat_w2 = v_w2 / (1. - self.beta2**t)
        vhat_b2 = v_b2 / (1. - self.beta2**t)
        vhat_w3 = v_w3 / (1. - self.beta2**t)
        vhat_b3 = v_b3 / (1. - self.beta2**t)
        
        w1 = w1 - self.alpha * mhat_w1 / (self.cls.sqrt(vhat_w1) + self.eps)
        b1 = b1 - self.alpha * mhat_b1 / (self.cls.sqrt(vhat_b1) + self.eps)
        w2 = w2 - self.alpha * mhat_w2 / (self.cls.sqrt(vhat_w2) + self.eps)
        b2 = b2 - self.alpha * mhat_b2 / (self.cls.sqrt(vhat_b2) + self.eps)
        w3 = w3 - self.alpha * mhat_w3 / (self.cls.sqrt(vhat_w3) + self.eps)
        b3 = b3 - self.alpha * mhat_b3 / (self.cls.sqrt(vhat_b1) + self.eps)
        
        return m_w1, m_b1, m_w2, m_b2, m_w3, m_b3, v_w1, v_b1, v_w2, v_b2, v_w3, v_b3, mhat_w1, mhat_b1, mhat_w2, mhat_b2, mhat_w3, mhat_b3, vhat_w1, vhat_b1, vhat_w2, vhat_b2, vhat_w3, vhat_b3, w1, b1, w2, b2, w3, b3

    # get prediction
    def __get_predictions__(self, a3):
        """
        Parameters
        ----------
        a3 : numpy.ndarray or cupy.ndarray
            output vector
        Returns
        -------
        prediction : numpy.ndarray or cupy.ndarray
            prediction
        """
        return self.cls.argmax(a3, 0)

    # get accuracy
    def get_accuracy(self, predictions, Y):
        """
        Parameters
        ----------
        prediction : numpy.ndarray or cupy.ndarray
            prediction
        Y : numpy.ndarray or cupy.ndarray
            labels
        Returns
        -------
        accuracy : float
            accuracy
        """
        return self.cls.sum(predictions == Y) / Y.size

    # get loss
    def get_loss(self, predictions, Y):
        """
        Parameters
        ----------
        prediction : numpy.ndarray or cupy.ndarray
            prediction
        Y : numpy.ndarray or cupy.ndarray
            labels
        Returns
        -------
        loss : float
            loss
        """
        return ( 0.5 * (predictions - Y)**2 ).sum() / Y.size

    # get ce loss
    def get_ce_loss(self, a, Y):
        """
        Parameters
        ----------
        prediction : numpy.ndarray or cupy.ndarray
            prediction
        Y : numpy.ndarray or cupy.ndarray
            labels
        Returns
        -------
        ce loss : float
            ce loss
        """
        return -self.cls.sum( self.cls.log( a[Y, range(a.shape[1])] ) ) / Y.size

    # evaluate prediction
    def predictions(self, X, w1, b1, w2, b2, w3, b3):
        """
        Parameters
        ----------
        X : numpy.ndarray or cupy.ndarray
            features
        w1 : numpy.ndarray or cupy.ndarray
            weights
        b1 : numpy.ndarray or cupy.ndarray
            biases
        w2 : numpy.ndarray or cupy.ndarray
            weights
        b2 : numpy.ndarray or cupy.ndarray
            biases
        w3 : numpy.ndarray or cupy.ndarray
            weights
        b3 : numpy.ndarray or cupy.ndarray
            biases
        Returns
        -------
        prediction : numpy.ndarray or cupy.ndarray
            prediction
        """
        _, _, _, _, _, a3 = self.__forward_propagation__(w1, b1, w2, b2, w3, b3, X)
        predictions = self.__get_predictions__(a3)
        return predictions