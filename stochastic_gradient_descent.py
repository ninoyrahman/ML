'''
Created on Feb 06, 2025

@author: ninoy
'''
from neural_network import NN

class stochastic_gradient_descent(NN):
    """
    class for stochastic gradient descent algorithm

    ...

    Attributes
    ----------

    Methods
    -------
    train(self, X, Y):
        train model
    """

    def __init__(self, label_number, 
                    alpha=0.1, 
                    epoch=500, 
                    activation='ReLU', 
                    layer_size=[784, 10, 10, 10], 
                    accuracy=0.9, 
                    batch_size=32, 
                    gradient_clip=None,
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
        gpu : bool
            flag for gpu usage
        """
        super().__init__(label_number, alpha=alpha, epoch=epoch, activation=activation, layer_size=layer_size, accuracy=accuracy, batch_size=batch_size, gradient_clip=gradient_clip, gpu=gpu)

    # conduct stochastic gradient descent
    def train(self, X, Y):
        """
        Parameters
        ----------
        X : numpy.ndarray or cupy.ndarray
            features
        Y : numpy.ndarray or cupy.ndarray
            label
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
        w1, b1, w2, b2, w3, b3 = self.initialze_parameters()

        for i in range(self.epoch):

            # shuffle data
            data = self.cls.c_[Y, X.T]
            self.cls.random.shuffle(data)
            data = data.T
            Y_new = self.cls.array(data[0, :], dtype=self.cls.int32)
            X_new = data[1:, :]

            for j in range(0, Y.size, self.batch_size):

                # select batch
                X_batch = X_new[:, j:j+self.batch_size]
                Y_batch = Y_new[j:j+self.batch_size]
            
                z1, a1, z2, a2, z3, a3 = self.__forward_propagation__(w1, b1, w2, b2, w3, b3, X_batch)
                dw1, db1, dw2, db2, dw3, db3 = self.__backward_propagation__(z1, a1, z2, a2, z3, a3, w1, w2, w3, X_batch, Y_batch)
                w1, b1, w2, b2, w3, b3 = self.__update_parameters__(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3)
            
            predictions = self.predictions(X, w1, b1, w2, b2, w3, b3)
            acc = self.get_accuracy(predictions, Y)
            if i % 100 == 0:
                print("Epoch: ", i, "Accuracy: ", acc)
            if acc > self.accuracy:
                print("Epoch: ", i, "Accuracy: ", acc)
                return w1, b1, w2, b2, w3, b3

        return w1, b1, w2, b2, w3, b3