"""
Logistic regression model
"""

import numpy as np
import math


class Logistic(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.threshold = 0.5  # To threshold the sigmoid
        self.weight_decay = weight_decay

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        return 1 / (1 + np.exp(-z))


    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.
        Train a logistic regression classifier for each class i to predict the probability that y=i

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        
        # one hot encode y_train - true label should be 1 and rest -1
        # onehot = N * number of classes

        ## check how dtype=int increases accuracy

        y_train_onehot = - np.ones((N, self.n_class), dtype=int)
        y_train_onehot[np.arange(N), y_train] = 1

        for epoch in range(self.epochs):
            # sigmoid = - y * w * x
            sigmoid = self.sigmoid(- y_train_onehot.T * (self.w @ X_train.T))
            # graident = (sigmoid * y * x)/N
            gradient = - ((sigmoid * y_train_onehot.T) @ X_train)/N
            # update weights and add regularization term
            self.w = self.w - self.lr * (gradient + self.weight_decay * self.w)

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        # return indices of predicted labels
        return np.argmax(self.sigmoid(self.w.dot(X_test.T)), axis=0)