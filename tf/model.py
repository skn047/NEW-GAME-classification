import tensorflow as tf
from tensorflow import keras
import numpy as np


class AlexNet(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(*kwargs)
        self.conv1 = keras.layers.Conv2D(filters=96, kernel_size=7, strides=[1, 1], activation="relu")
        self.pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv2 = keras.layers.Conv2D(filters=128, kernel_size=4, strides=[1, 1], activation="relu")
        self.pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.fc1 = keras.layers.Dense(150)
        self.fc2 = keras.layers.Dense(6)
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, input, training=True):
        x = self.bn1(self.pool1(self.conv1(input)), training=training)
        x = self.bn2(self.pool2(self.conv2(x)), training=training)
        x = tf.reshape(x, [input.shape[0], -1])
        x = self.fc1(x)
        output = self.fc2(x)

        return output
