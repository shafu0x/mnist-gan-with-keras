import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn import preprocessing

INPUT_LAYER = 784
HIDDEN_LAYER_1 = 512
HIDDEN_LAYER_2 = 216
HIDDEN_LAYER_3 = 64
HIDDEN_LAYER_4 = 32
HIDDEN_LAYER_5 = 2

OPTIMIZER = 'adam'
LOSS_FUNCTION = 'kullback_leibler_divergence'
ACTIVATION_FUNCTION = 'sigmoid'

class Discriminator:
    """Discriminator.
    A simple 6 layered Neural Network (NN) used to detect images generated from
    the generator and images from the MNIST dataset.
    """
    def __init__(self, batch_size=128, epochs=2):
        """Initializes a Sequential NN with the keras Sequential object.
        It takes as input one image from the MNIST dataset (shape=(None, 784))
        and returns one of 1 if it thinks its from the dataset and 0 if doesn't
        think so.

        Args:
            batch_size (int, optional): Batch size while training. Defaults to
                128.
            epochs (int, optional): Number of epochs to train. Defaults to 2.
        """
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = Sequential()
        self.model.add(Dense(units=HIDDEN_LAYER_1, activation=ACTIVATION_FUNCTION, input_dim=INPUT_LAYER))
        self.model.add(Dense(units=HIDDEN_LAYER_2, activation=ACTIVATION_FUNCTION))
        self.model.add(Dense(units=HIDDEN_LAYER_3, activation=ACTIVATION_FUNCTION))
        self.model.add(Dense(units=HIDDEN_LAYER_4, activation=ACTIVATION_FUNCTION))
        self.model.add(Dense(units=HIDDEN_LAYER_5, activation='softmax'))

        self.model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])

    def train(self, X, y, verbose=True):
        """Trains the NN.
        Training is always done on the whole normalized training set of one
        generation. The labels are the actions that where taken.

        Args:
            X (np.array): Image (Shape=(None, 784)).
            y (np.array): Label (Shape=(None, 2)).
        """
        verbosity = 1
        if not verbose:
            verbosity = 0

        # normalize data.
        X = preprocessing.normalize(X)

        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=verbosity)

    def prediction(self, X):
        """Predicts if image is from the MNIST dataset or not.

        Args:
            X (np.array): Image (Shape=(None, 784)).

        Returns:
            is_from_mnist (int): 1 if true and 0 otherwise.
        """
        # otherwise keras has problems.
        X = X.reshape(1, INPUT_LAYER)

        prediction = self.model.predict(X)

        if np.argmax(prediction) == 0:
            is_from_mnist = 0
        else:
            is_from_mnist = 1

        return is_from_mnist

    def save(self, name):
        """Saves the model.
        Args:
            name (str): Name of the model.
        """
        # this is where are models are saved.
        FULL_PATH = '/Users/Shafou/Desktop/gan/discriminator/'
        self.model.save(FULL_PATH + name)

    def eval(self, X, y):
        """Returns the accuracy of the NN on the test data.
        Args:
            X (np.array): Image (Shape=(None, 784)).
            y (np.array): Label (Shape=(None, 2)).

        Returns:
            accuracy (float): Accuracy on the test set.
        """
        accuracy = []

        #predictions = self.prediction(X)
        print(X.shape)
        predictions = self.model.predict(X)

        print(predictions.shape)

        for i in range(len(predictions)):
            if np.argmax(y[i]) == np.argmax(predictions[i]):
                accuracy.append(1)

            else:
                accuracy.append(0)


        return sum(accuracy) / float(len(accuracy))
