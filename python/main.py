from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
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

# Parameters
n_inputs = 28*28 # MNIST image shape 28*28
n_outputs = 10  # MNIST classes from 0-9 digits

n_hiddens = 256  # number of neurons

n_batches = np.shape(trainX)[1] # 55000 number of samples for training
train_epochs = 15
warm_epochs = 10

beta = 5.0
gamma = 5.0


# Initial Model
model = ADMM_NN(n_inputs, n_hiddens, n_outputs, n_batches)

# Warming Model
model.warming(trainX, trainY, warm_epochs, beta, gamma)

# Training Model
list_loss_train = []
list_loss_valid = []
list_accuracy_train = []
list_accuracy_valid = []
for i in range(train_epochs):
    print("------ Training: {:d} ------".format(i))
    loss_train, accuracy_train = model.fit(trainX, trainY, beta, gamma)
    loss_valid, accuracy_valid = model.evaluate(validX, validY)

    print("Loss train: %3f, accuracy train: %3f" % (np.array(loss_train), np.array(accuracy_train)))
    print("Loss valid: %3f, accuracy valid: %3f" % (np.array(loss_valid), np.array(accuracy_valid)))

    # Append loss and accuracy
    list_loss_train.append(np.array(loss_train))
    list_loss_valid.append(np.array(loss_valid))
    list_accuracy_train.append(np.array(accuracy_train))
    list_accuracy_valid.append(np.array(accuracy_valid))

    # Drawing loss, accuracy of train and valid
    model.drawcurve(list_loss_train, list_loss_valid, 1, 'loss_train', 'loss_valid')
    model.drawcurve(list_accuracy_train, list_accuracy_valid, 2, 'acc_train', 'acc_valid')

# Evaluate Model on test set
loss, accuracy = model.evaluate(testX, testY)
print("Loss valid: %3f, accuracy valid: %3f" % (np.array(loss), np.array(accuracy)))
