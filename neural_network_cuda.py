'''
Created on Feb 06, 2025

@author: ninoy
'''
import numpy as np
import numba as nb
import math
from numba import cuda, float64

# parameter
label_number = 10
epoch=500
alpha=0.1
activation='ReLU'
layer_size=[784, 10, 10, 10]
accuracy=0.9
batch_size=32
gradient_clip=None
gamma=0.9 
beta=0.9 
beta1=0.9
beta2=0.999
sample_size = 100

# cuda parameter
TPB = 32
threadsperblock = (TPB, TPB)

# increment
@cuda.jit
def increment(Z, B):
    """Perform increment Z += B."""
    i, j = cuda.grid(2)
    if i < Z.shape[0] and j < Z.shape[1]:
        Z[i, j] += B[i, 1]

# ReLU
@cuda.jit
def ReLU(A, Z):
    """Perform ReLU"""
    i, j = cuda.grid(2)
    if i < Z.shape[0] and j < Z.shape[1]:
        A[i, j] = max(Z[i, j], 0)

# softmax
@cuda.jit
def expZ(A, Z):
    """Perform A = exp(Z)."""
    i, j = cuda.grid(2)
    if i < Z.shape[0] and j < Z.shape[1]:
        A[i, j] = math.exp(Z[i, j])   

# matrix transpose
@cuda.jit
def transpose(AT, A):
    """Perform transpose."""
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        AT[i, j] = A[j, i]

# matrix multiplication
@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B."""
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B using CUDA shared memory.

    Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float64)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float64)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float64(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < B.shape[1] and (ty + i * TPB) < B.shape[0]:
            sB[ty, tx] = B[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp

# weight and biases initialization
def initialze():
    np.random.seed(42)
    w1 = np.random.normal(size=(layer_size[1], layer_size[0])).astype(np.float64) * np.sqrt(1. / layer_size[0])
    b1 = np.random.normal(size=(layer_size[1], 1)).astype(np.float64) * np.sqrt(1. / layer_size[1])
    w2 = np.random.normal(size=(layer_size[2], layer_size[1])).astype(np.float64) * np.sqrt(1. / ( layer_size[2] * 2. ))
    b2 = np.random.normal(size=(layer_size[2], 1)).astype(np.float64) * np.sqrt(1. / layer_size[2])
    w3 = np.random.normal(size=(layer_size[3], layer_size[2])).astype(np.float64) * np.sqrt(1. / ( layer_size[3] * 2. ))
    b3 = np.random.normal(size=(layer_size[3], 1)).astype(np.float64) * np.sqrt(1. / layer_size[3])
    
    z1 = np.zeros((layer_size[1], sample_size), dtype=np.float64)
    a1 = np.zeros((layer_size[1], sample_size), dtype=np.float64)
    z2 = np.zeros((layer_size[2], sample_size), dtype=np.float64)
    a2 = np.zeros((layer_size[2], sample_size), dtype=np.float64)
    z3 = np.zeros((layer_size[3], sample_size), dtype=np.float64)
    a3 = np.zeros((layer_size[3], sample_size), dtype=np.float64)
    
    dw1 = np.zeros((layer_size[1], layer_size[0]), dtype=np.float64)
    db1 = np.zeros((layer_size[1], 1), dtype=np.float64)
    dw2 = np.zeros((layer_size[2], layer_size[1]), dtype=np.float64)
    db2 = np.zeros((layer_size[2], 1), dtype=np.float64)
    dw3 = np.zeros((layer_size[3], layer_size[2]), dtype=np.float64)
    db3 = np.zeros((layer_size[3], 1), dtype=np.float64)

    return w1, b1, w2, b2, w3, b3, z1, a1, z2, a2, z3, a3, dw1, db1, dw2, db2, dw3, db3

# print model parameters
def print_parameter():
    print('')
    print('NN parameters:')
    print('sample size        = ', sample_size)
    print('number of labels   = ', label_number)
    print('epoch              = ', epoch)
    print('learning_rate      = ', alpha)
    print('activation         = ', activation)
    print('layer size         = ', layer_size)
    print('accuracy           = ', accuracy)
    print('batch size for SGD = ', batch_size)
    print('gradient clip      = ', gradient_clip)
    print('gamma              = ', gamma)
    print('beta               = ', beta)
    print('beta1              = ', beta1)
    print('beta2              = ', beta2)
    # print('blockspergrid      = ', blockspergrid)
    print('threadsperblock    = ', threadsperblock)    
    print('')        

    # # activation function ReLU/sigmoid
    # def factivation(self, z):
    #     if self.__activation == 'ReLU':
    #         return np.maximum(z, 0)
    #     else:
    #         return 1.0 / (1.0 + np.exp(-z))

    # # derivative of activation function ReLU/sigmoid
    # def dfactivation(self, z):
    #     if self.__activation == 'ReLU':
    #         return z > 0
    #     else:
    #         return self.factivation(z) * (1.0 - self.factivation(z))

    # # softmax function at output layer
    # def softmax(self, z):
    #     return np.exp(z) / sum(np.exp(z))

    # # softmax function at output layer
    # def dsoftmax(self, z):
    #     return self.softmax(z) * ( 1.0 - self.softmax(z) )


# error
@cuda.jit
def error_3(E, A, Y, m):
    """Perform A = exp(Z)."""
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < A.shape[1]:
        E[i, j] = (A[i, j] - Y[i, j]) / m

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, label_number))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# derivative of activation function ReLU
def dfactivation(z):
    return z > 0

# @cuda.jit
def gd(z1, a1, z2, a2, z3, a3, w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, X, Y):

    # host to device
    d_z1 = cuda.to_device(z1)
    d_a1 = cuda.to_device(a1)
    d_z2 = cuda.to_device(z2)
    d_a2 = cuda.to_device(a2)
    d_z3 = cuda.to_device(z3)
    d_a3 = cuda.to_device(a3)
    
    d_w1 = cuda.to_device(w1)
    d_b1 = cuda.to_device(b1)
    d_w2 = cuda.to_device(w2)
    d_b2 = cuda.to_device(b2)
    d_w3 = cuda.to_device(w3)
    d_b3 = cuda.to_device(b3)

    # d_X = cuda.to_device(X)
    # # d_Y = cuda.to_device(Y)

    # # forward propagation    
    # # hidden layer 1
    # blockspergrid_x = math.ceil(layer_size[1] / threadsperblock[0])
    # blockspergrid_y = math.ceil(sample_size / threadsperblock[1])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    # # print(blockspergrid, threadsperblock)
    # fast_matmul[blockspergrid, threadsperblock](d_w1, d_X, d_z1)
    # increment[blockspergrid, threadsperblock](d_z1, d_b1)
    # ReLU[blockspergrid, threadsperblock](d_a1, d_z1)

    # z1 = d_z1.copy_to_host()
    # a1 = d_a1.copy_to_host()

    # # hidden layer 2
    # blockspergrid_x = math.ceil(layer_size[2] / threadsperblock[0])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    # # print(blockspergrid, threadsperblock)
    # fast_matmul[blockspergrid, threadsperblock](d_w2, d_a1, d_z2)
    # increment[blockspergrid, threadsperblock](d_z2, d_b2)
    # ReLU[blockspergrid, threadsperblock](d_a2, d_z2)
    
    # z2 = d_z2.copy_to_host()
    # a2 = d_a2.copy_to_host()

    # # output layer
    # blockspergrid_x = math.ceil(layer_size[3] / threadsperblock[0])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    # # print(blockspergrid, threadsperblock)
    # fast_matmul[blockspergrid, threadsperblock](d_w3, d_a2, d_z3)
    # increment[blockspergrid, threadsperblock](d_z3, d_b3)
    # expZ[blockspergrid, threadsperblock](d_a3, d_z3)
    
    # z3 = d_z3.copy_to_host()
    # a3 = d_a3.copy_to_host()
    # a3 = a3 / np.sum(a3)
    # d_a3 = cuda.to_device(a3)

    # # backward propagation
    # m = Y.size
    # one_hot_Y = one_hot(Y)

    # # copy to device
    # d_z1 = cuda.to_device(z1)
    # d_a1 = cuda.to_device(a1)
    # d_z2 = cuda.to_device(z2)
    # d_a2 = cuda.to_device(a2)
    # d_z3 = cuda.to_device(z3)
    # d_a3 = cuda.to_device(a3)
    # d_Y  = cuda.to_device(one_hot_Y)

    # d_dw1 = cuda.to_device(dw1)
    # d_db1 = cuda.to_device(db1)
    # d_dw2 = cuda.to_device(dw2)
    # d_db2 = cuda.to_device(db2)
    # d_dw3 = cuda.to_device(dw3)
    # d_db3 = cuda.to_device(db3)    

    # # output layer to hidden layer 2
    # blockspergrid_x = math.ceil(layer_size[3] / threadsperblock[0])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)    
    
    # # delta = (1.0 / m) * (a3 - one_hot_Y)
    # delta = np.zeros_like(a3)
    # d_delta = cuda.to_device(delta)
    # error_3[blockspergrid, threadsperblock](d_delta, d_a3, d_Y, m)
    # delta = d_delta.copy_to_host()
    
    # # dw3 = delta.dot(a2.T)
    # a2T = np.zeros_like(a2)
    # d_a2T = cuda.to_device(a2T)
    # transpose[blockspergrid, threadsperblock](d_a2T, d_a2)
    # fast_matmul[blockspergrid, threadsperblock](d_delta, d_a2T, d_dw3)
    # dw3 = d_dw3.copy_to_host()
    
    # db3 = np.sum(delta)

    # # hidden layer 2 to hidden layer 1
    # delta1 = w3.T.dot(delta) * dfactivation(z2)
    # dw2 = delta1.dot(a1.T)
    # db2 = np.sum(delta1)

    # # hidden layer 1 to input layer
    # delta2 = w2.T.dot(delta1) * dfactivation(z1)
    # dw1 = delta2.dot(X.T)
    # db1 = np.sum(delta2)    
        
    return z1, a1, z2, a2, z3, a3, dw1, db1, dw2, db2, dw3, db3
