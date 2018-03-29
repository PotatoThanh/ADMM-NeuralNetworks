from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe
from ADMM_NN import ADMM_NN
from tensorflow.examples.tutorials.mnist import input_data

tfe.enable_eager_execution()

# Load MNIST data
mnist = input_data.read_data_sets("./data/", one_hot=True)

trainX = np.transpose(mnist.train.images).astype(np.float32)
trainY = np.transpose(mnist.train.labels).astype(np.float32)

validX = np.transpose(mnist.validation.images).astype(np.float32)
validY = np.transpose(mnist.validation.labels).astype(np.float32)

testX = np.transpose(mnist.test.images).astype(np.float32)
testY = np.transpose(mnist.test.labels).astype(np.float32)

# Network Parameters
n_inputs = 28*28 # MNIST image shape 28*28
n_outputs = 10  # MNIST classes from 0-9 digits
n_hiddens = 256  # number of neurons
n_batches   = 5000
epochs = 10
beta = 5.0
gamma = 5.0

model = ADMM_NN(n_inputs, n_hiddens, n_outputs, n_batches)
model.fit(validX, validY, epochs, beta, gamma)
# Parameters


# Warming up

# Training Neural Network

# Validation

# x = np.array([[2.0,20]])
# m = x/2
# print(m)