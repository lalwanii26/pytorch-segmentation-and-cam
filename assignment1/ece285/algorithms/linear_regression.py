"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        
        # one hot encode y_train - true label should be 1 and rest 0
        # onehot = N * number of classes    
        y_train_onehot = - np.zeros((N, self.n_class), dtype=int)
        y_train_onehot[np.arange(N), y_train] = 1

        for epoch in range(self.epochs):
            # calculate y_pred =  X * w
            y_pred = np.dot(X_train, self.w.T)
            # calculate loss = y_pred - y
            loss = y_pred - y_train_onehot
            # calculate gradient = 2 (X * loss)/N
            gradient = 2 * np.dot(X_train.T, loss) / N
            # add regularization term - lambda/2 * w^2
            gradient += self.weight_decay * self.w.T
            # update weights -> w = w - lr * gradient 
            self.w -= self.lr * gradient.T
            # print(self.w)

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
        # return max value of number of classes - row
        return np.argmax(np.dot(X_test,self.w.T), axis=1)
       