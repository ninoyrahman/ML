# import data and libs
import numpy as np
import cupy as cp
import pandas as pd
import time
import sys

from gradient_descent import gradient_descent
from stochastic_gradient_descent import stochastic_gradient_descent
from adam import adam

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

# read mnist data
data = pd.read_csv('data/mnist.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

# testing data
test_size = 20000 
data_test = data[0:test_size].T
Y_test = data_test[0]
X_test = data_test[1:n] / 255.

# training data
data_train = data[test_size:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
_,m_train = X_train.shape

X_train_d = cp.asarray(X_train)
Y_train_d = cp.asarray(Y_train)
X_test_d = cp.asarray(X_test)
Y_test_d = cp.asarray(Y_test)

print('training size : ', X_train.shape, Y_train.shape)
print('test size     : ', X_test.shape,  Y_test.shape)

layer_size=[784, 2048, 1024, 10]
print('layer size    : ', layer_size)

# gradient descent - neural network
print('')
print('gradient descent')
nn = gradient_descent(label_number=10, alpha=0.01, epoch=500, activation='ReLU', layer_size=layer_size)
nn.print_parameter()
t = time.process_time()
W1, b1, W2, b2, W3, b3 = nn.train(X_train_d, Y_train_d)
elapsed_time = time.process_time() - t
dev_predictions = nn.predictions(X_test_d, W1, b1, W2, b2, W3, b3)
print("Accuracy: ", nn.get_accuracy(dev_predictions, Y_test_d), "Loss: ", nn.get_loss(dev_predictions, Y_test_d))
print('Elapsed time(GD): ', elapsed_time)

# stochastic gradient descent - neural network
print('')
print('stochastic gradient descent')
nn = stochastic_gradient_descent(label_number=10, alpha=0.01, epoch=500, activation='ReLU', layer_size=layer_size, accuracy=0.9, batch_size=32, gradient_clip=1.0)
nn.print_parameter()
t = time.process_time()
W1, b1, W2, b2, W3, b3 = nn.train(X_train_d, Y_train_d)
elapsed_time = time.process_time() - t
dev_predictions = nn.predictions(X_test_d, W1, b1, W2, b2, W3, b3)
print("Accuracy: ", nn.get_accuracy(dev_predictions, Y_test_d), "Loss: ", nn.get_loss(dev_predictions, Y_test_d))
print('Elapsed time(SGD): ', elapsed_time)

# adam - neural network
print('')
print('adam')
nn = adam(label_number=10, alpha=0.01, epoch=500, activation='ReLU', layer_size=layer_size, accuracy=0.9, batch_size=Y_train.size, gradient_clip=1.0, beta1=0.9, beta2=0.999)
nn.print_parameter()
t = time.process_time()
W1, b1, W2, b2, W3, b3 = nn.train(X_train_d, Y_train_d)
elapsed_time = time.process_time() - t
dev_predictions = nn.predictions(X_test_d, W1, b1, W2, b2, W3, b3)
print("Accuracy: ", nn.get_accuracy(dev_predictions, Y_test_d), "Loss: ", nn.get_loss(dev_predictions, Y_test_d))
print('Elapsed time(adam): ', elapsed_time)

sys.stdout = orig_stdout
f.close()