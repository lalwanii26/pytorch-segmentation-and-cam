"""
K Nearest Neighbours Model
"""
import numpy as np


class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        # TODO: implement me
       
        #  Use the contained training set to predict labels for test samples
        predict = []
        for i in range(x_test.shape[0]):
            # get the index of the k nearest neighbours
            index = np.argsort(distance[i])[:k_test]
            # get the labels of the k nearest neighbours
            label = self._y_train[index]
            # find the most common label
            predict.append(np.argmax(np.bincount(label)))
        
        return np.array(predict)
    
        pass

    def calc_dis_one_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """

        # TODO: implement me
         # assign x_test.shape[0] to a  new variable
        N = x_test.shape[0]
        # asisgn self._x_train.shape[0] to a new variable
        D = self._x_train.shape[0]
        # initialize a new array with shape (N, D)
        distance = np.zeros((N, D))
        # use one loop to iterate through test samples and training samples to calculate distance
        for i in range(N):
            distance[i] = np.sqrt(np.sum((x_test[i] - self._x_train) ** 2, axis=1))

        return distance

    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
         # assign x_test.shape[0] to a  new variable
        N = x_test.shape[0]
        # asisgn self._x_train.shape[0] to a new variable
        D = self._x_train.shape[0]
        # initialize a new array with shape (N, D)
        distance = np.zeros((N, D))
        # use two loops to iterate through test samples and training samples to calculate distance
        for i in range(N):
            for j in range(D):
                distance[i, j] = np.sqrt(np.sum((x_test[i] - self._x_train[j]) ** 2))

        return distance