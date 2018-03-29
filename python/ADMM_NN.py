from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()

class ADMM_NN(object):
    """ Class for ADMM Neural Network. """

    def __init__(self, n_inputs, n_hiddens, n_outputs, n_batches):

        """
        Initialize variables for NN.
        Raises:
            ValueError: Column input samples, for example, the input size of MNIST data should be (28x28, *) instead of (*, 28x28).
        :param n_inputs: Number of inputs.
        :param n_hiddens: Number of hidden units.
        :param n_outputs: Number of outputs
        :param n_batches: Number of data sample that you want to train
        :param return:
        """
        self.a0 = np.zeros((n_inputs, n_batches))

        self.w1 = np.zeros((n_hiddens, n_inputs))
        self.w2 = np.zeros((n_hiddens, n_hiddens))
        self.w3 = np.zeros((n_outputs, n_hiddens))

        self.z1 = np.random.rand(n_hiddens, n_batches)
        self.a1 = np.random.rand(n_hiddens, n_batches)

        self.z2 = np.random.rand(n_hiddens, n_batches)
        self.a2 = np.random.rand(n_hiddens, n_batches)

        self.z3 = np.random.rand(n_outputs, n_batches)

        self.lambda_larange = np.ones((n_outputs, n_batches))

    def _relu(self, x):
        """
        Relu activation function
        :param x: input x
        :return: max 0 and x
        """
        return tf.maximum(0.0,x)

    def _weight_update(self, layer_output, activation_input):
        """
        Consider it now the minimization of the problem with respect to W_l.
        For each layer l, the optimal solution minimizes ||z_l - W_l a_l-1||^2. This is simply
        a least square problem, and the solution is given by W_l = z_l p_l-1, where p_l-1
        represents the pseudo-inverse of the rectangular activation matrix a_l-1.
        :param layer_output: output matrix (z_l)
        :param activation_input: activation matrix l-1  (a_l-1)
        :return: weight matrix
        """
        pinv = np.linalg.pinv(activation_input)
        # pinv = tf.matmul(tf.matrix_inverse(tf.matmul(tf.matrix_transpose(activation_input), activation_input)),tf.matrix_transpose(activation_input))
        weight_matrix = tf.matmul(tf.cast(layer_output, tf.float32), tf.cast(pinv, tf.float32))
        return weight_matrix

    def _activation_update(self, next_weight, next_layer_output, layer_nl_output, beta, gamma):
        """
        Minimization for a_l is a simple least squares problem similar to the weight update.
        However, in this case the matrix appears in two penalty terms in the problem, and so
        we must minimize:
            beta ||z_l+1 - W_l+1 a_l||^2 + gamma ||a_l - h(z_l)||^2
        :param next_weight:  weight matrix l+1 (w_l+1)
        :param next_layer_output: output matrix l+1 (z_l+1)
        :param layer_nl_output: activate output matrix h(z) (h(z_l))
        :param beta: value of beta
        :param gamma: value of gamma
        :return: activation matrix
        """
        # Calculate ReLU
        layer_nl_output = self._relu(layer_nl_output)

        # Activation inverse
        m1 = beta*tf.matmul(tf.matrix_transpose(next_weight), next_weight)
        m2 = tf.scalar_mul(gamma, tf.eye(tf.cast(m1.get_shape()[0], tf.int32)))
        av = tf.matrix_inverse(tf.cast(m1, tf.float32) + tf.cast(m2, tf.float32))

        # Activation formulate
        m3 = beta*tf.matmul(tf.matrix_transpose(next_weight), next_layer_output)
        m4 = gamma * layer_nl_output
        af = tf.cast(m3, tf.float32) + tf.cast(m4, tf.float32)

        # Output
        return tf.matmul(av, af)

    def _argminz(self, a, w, a_in, beta, gamma):
        """
        This problem is non-convex and non-quadratic (because of the non-linear term h).
        Fortunately, because the non-linearity h works entry-wise on its argument, the entries
        in z_l are decoupled. This is particularly easy when h is piecewise linear, as it can
        be solved in closed form; common piecewise linear choices for h include rectified
        linear units (ReLUs), that its used here, and non-differentiable sigmoid functions.
        :param a: activation matrix (a_l)
        :param w:  weight matrix (w_l)
        :param a_in: activation matrix l-1 (a_l-1)
        :param beta: value of beta
        :param gamma: value of gamma
        :return: output matrix
        """
        m = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        sol1 = (gamma*a + beta*m)/(gamma + beta)
        sol2 = m
        z1 = np.zeros_like(a)
        z2 = np.zeros_like(a)
        z  = np.zeros_like(a)

        tf.where(sol1>=0.0, z1, sol1)
        tf.where(sol2<=0.0, z2, sol2)

        fz_1 = tf.square(gamma * (a - self._relu(z1))) + beta * (tf.square(z1 - m))
        fz_2 = tf.square(gamma * (a - self._relu(z2))) + beta * (tf.square(z2 - m))

        index_z1 = fz_1<=fz_2
        index_z2 = fz_2<fz_1

        tf.where(index_z1, z, z1)
        tf.where(index_z2, z, z2)
        return z

    def _argminlastz(self, targets, eps, w, a_in, beta):
        """
        Minimization of the last output matrix, using the above function.
        :param targets: target matrix (equal dimensions of z) (y)
        :param eps: lagrange multiplier matrix (equal dimensions of z) (lambda)
        :param w: weight matrix (w_l)
        :param a_in: activation matrix l-1 (a_l-1)
        :param beta: value of beta
        :return: output matrix last layer
        """
        m = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        z = (targets - eps + beta*m)/(1+beta)
        return z

    def _lambda_update(self, zl, w, a_in, beta):
        """
        Lagrange multiplier update.
        :param zl: output matrix last layer (z_L)
        :param w: weight matrix last layer (w_L)
        :param a_in: activation matrix l-1 (a_L-1)
        :param beta: value of beta
        :return: lagrange update
        """
        mpt = tf.matmul(tf.cast(w, tf.float32), tf.cast(a_in, tf.float32))
        lambda_up = beta*(zl-mpt)
        return lambda_up

    def feed_forward(self, inputs):
        """
        Calculate feed forward pass for neural network
        :param inputs: inputs features
        :return: value of prediction
        """
        outputs = self._relu(tf.matmul(self.w1, inputs))
        outputs = self._relu(tf.matmul(self.w2, outputs))
        outputs = tf.matmul(self.w3, outputs)
        return outputs

    def fit(self, inputs, labels, epochs, beta, gamma):
        """
        Training ADMM Neural Network by minimizing sub-problems
        :param inputs: input of training data samples
        :param outputs: label of training data samples
        :param epochs: number of epochs
        :param beta: value of beta
        :param gamma: value of gamma
        :return: loss value
        """
        self.a0 = inputs
        for i in range(epochs):
            print(i)

            # Input layer
            self.w1 = self._weight_update(self.z1, self.a0)
            self.a1 = self._activation_update(self.w2, self.z2, self.z1, beta, gamma)
            self.z1 = self._argminz(self.a1, self.w1, self.a0, beta, gamma)

            # Hidden layer
            self.w2 = self._weight_update(self.z2, self.a1)
            self.a2 = self._activation_update(self.w3, self.z3, self.z2, beta, gamma)
            self.z2 = self._argminz(self.a2, self.w2, self.a1, beta, gamma)

            # Output layer
            self.w3 = self._weight_update(self.z3, self.a2)
            self.z3 = self._argminlastz(labels, self.lambda_larange, self.w3, self.a2, beta)
            self.lambda_larange = self._lambda_update(self.z3, self.w3, self.a2, beta)

            forward = self.feed_forward(inputs)
            loss_train = tf.reduce_mean(tf.square(forward - labels))
            # accuracy = np.zeros_like(forward)
            # tf.where(tf.argmax(forward, axis=1), accuracy, np.ones_like(forward))
            # accuracy = tf.reduce_sum(accuracy)
            print(loss_train)
            # print(accuracy)
        return loss_train

    # def predict(self, inputs):
    #
    #
    #
    # def fit(self, inputs, outputs,):
    #
    # def evaluate(self, inputs, outputs, isCategrories = False ):

