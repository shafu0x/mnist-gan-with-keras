"""
This class represents the gan.
The gan is build from 2 networks. First the generator and secondly the
discriminator.

Training process:
1) Take the discriminator seperately and train it on the dataset and the data
    created by the generator.
2) Take the gan and set the discriminator weights as untrainable to train the
    generator.
3) Repeat this process.
"""
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import *
import numpy as np
from sklearn import preprocessing

import params
from generator import Generator
from discriminator import Discriminator

class Gan:
    """Gan (Generative adverserial network).
    A Gan made up of a generator and a discriminator.
    """
    def __init__(self, generator, discriminator):
        """
        Args:
            generator (keras.models.Sequential): The generator.
            discriminator (keras.models.Sequential): The discriminator.
        """
        self.generator = generator
        self.discriminator = discriminator

        self.layers = self.get_layers()
        self.generator_layers, self.discriminator_layers = self.get_generator_discriminator_layers()

        self.model = Sequential(self.get_layers())
        self.model.compile(loss=params.GAN_LOSS, optimizer=params.GAN_OPTIMIZER, metrics=['accuracy'])

    def get_layers(self):
        """Returns the layers of the generator and the discriminator in order.

        Returns:
            layers (list): List of the layers of the generator and the discriminator.
        """
        layers = []
        for generator_layer in self.generator.get_layers():
            layers.append(generator_layer)
        for discriminator_layer in self.discriminator.get_layers():
            layers.append(discriminator_layer)

        return layers

    def get_generator_discriminator_layers(self):
        """
        Return:
            generator_layers (list): Generator layers.
            discriminator_layers (list): Discriminator layers.
        """
        n_generator_layers = len(self.generator.get_layers())
        generator_layers = self.get_layers()[0: n_generator_layers]

        n_discriminator_layers = len(self.discriminator.get_layers())
        discriminator_layers = self.get_layers()[n_generator_layers: n_generator_layers+n_discriminator_layers]

        return generator_layers, discriminator_layers

    def set_discriminator_trainability(self, trainable):
        """
        Changes the trainability of the discriminator layers.

        Args:
            trainable (bool): Trainable if true. Not trainable if false.
        """
        for layer in self.discriminator_layers:
            layer.trainable = trainable

        # after changing the trainable param the model has to be compiled again.
        self.model.compile(loss=params.GAN_LOSS, optimizer=params.GAN_OPTIMIZER, metrics=['accuracy'])

    def train_generator(self, X, y, verbose=True):
        """Trains the generator.

        Generated images are passed through the generator and discriminator. The
        generator should create images like the actual data. The target is to
        fool the discriminator.

        Args:
            X (np.array): Generated image (Shape=(None, 16)).
            y (np.array): Label (Shape=(None, 2)).
        """
        self.set_discriminator_trainability(False)

        verbosity = 1
        if not verbose:
            verbosity = 0

        # normalize data.
        #X = preprocessing.normalize(X)

        self.model.fit(X, y, epochs=params.GAN_GENERATOR_TRAINING_EPOCH, batch_size=params.GAN_GENERATOR_TRAINING_BATCH_SIZE, verbose=verbosity)

        generator_layers = self.model.layers[0: 5]

        generator = Sequential(generator_layers)
        return generator

    def show_trainable(self):
        """
        Just a helper method to see the trainability of the layers.
        """
        for layer in self.model.layers:
            print(layer.trainable)
