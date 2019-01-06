import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn import preprocessing

import params

class Generator:
    """Generator.
    A simple 6 layered Neural Network (NN) used to generate images that look
    like images from the MNIST dataset using as input a normal shaped array of
    size 16. Cannot be trained.
    """
    def __init__(self):
        """Initializes a Sequential NN with the keras Sequential object.
        It takes as input a normal shaped array of size 16 and returns an array
        of shape=(None, 784).
        """

        self.model = Sequential()
        self.model.add(Dense(units=params.GENERATOR_HIDDEN_LAYER_1, activation=params.GENERATOR_ACTIVATION, input_dim=params.GENERATOR_INPUT_LAYER))
        self.model.add(Dense(units=params.GENERATOR_HIDDEN_LAYER_2, activation=params.GENERATOR_ACTIVATION))
        self.model.add(Dense(units=params.GENERATOR_HIDDEN_LAYER_3, activation=params.GENERATOR_ACTIVATION))
        self.model.add(Dense(units=params.GENERATOR_HIDDEN_LAYER_4, activation=params.GENERATOR_ACTIVATION))
        self.model.add(Dense(units=params.GENERATOR_HIDDEN_LAYER_5, activation=params.GENERATOR_ACTIVATION))

        #self.model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])

    def predict(self, X):
        """Predicts if image is from the MNIST dataset or not.

        Args:
            X (np.array): Image (Shape=(None, 16)).

        Returns:
            pred_y (np.array): Array of shape=(None, 784) that should look like
            an image from the MNIST dataset.
        """
        pred_y = self.model.predict(X)

        return pred_y

    def get_weights(self):
        """Reurns all the weights of the model.

        Returns:
            weights (list): All weights of the model.
        """

        return self.model.get_weights()

    def save(self, name):
        """Saves the model.

        Args:
            name (str): Name of the model.
        """
        # this is where are models are saved.
        FULL_PATH = '/Users/Shafou/Desktop/gan/discriminator/'
        self.model.save(FULL_PATH + name)
