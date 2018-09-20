import random

from modules.module import Operation
from tensorflow import keras

class Dense(Operation):

    units = 0
    activation = "ReLU"
    bias = True
    compatibility = ["Dense"]

    def __init__(self, ID, units, activation="ReLU", bias=True):
        super().__init__(ID, [type(self)])

        self.units = units
        self.activation = activation
        self.bias = bias

    def to_keras(self):
        return keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.bias
        )


class DenseS(Dense):
    min_units = 10
    max_units = 30

    def __init__(self, activation="ReLU", bias=True):
        super().__init__(
            ID="DenseS",
            units=random.randint(self.min_units, self.max_units)
        )


class DenseM(Dense):
    min_units = 30
    max_units = 100

    def __init__(self, activation="ReLU", bias=True):
        super().__init__(
            ID="DenseM",
            units=random.randint(self.min_units, self.max_units)
        )


class DenseL(Dense):
    min_units = 100
    max_units = 400

    def __init__(self, activation="ReLU", bias=True):
        super().__init__(
            ID="DenseL",
            units=random.randint(self.min_units, self.max_units)
        )