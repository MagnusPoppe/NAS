import random

from modules.operation import Operation
from tensorflow import keras

class Dense(Operation):

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
    _min_units = 10
    _max_units = 30

    def __init__(self):
        super().__init__(
            ID="DenseS",
            units=random.randint(self._min_units, self._max_units)
        )


class DenseM(Dense):
    _min_units = 30
    _max_units = 100

    def __init__(self):
        super().__init__(
            ID="DenseM",
            units=random.randint(self._min_units, self._max_units)
        )


class DenseL(Dense):
    _min_units = 100
    _max_units = 400

    def __init__(self):
        super().__init__(
            ID="DenseL",
            units=random.randint(self._min_units, self._max_units)
        )