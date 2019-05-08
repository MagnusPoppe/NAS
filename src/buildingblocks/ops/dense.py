import random
import time

from src.buildingblocks.ops.operation import Operation

counter = 0

class Dense(Operation):

    def __init__(self, ID, units, activation="relu", bias=True, dropout=True, dropout_probability=0.3):
        global counter
        if not ID:
            ID = "{}_{}".format(ID, counter)
            counter += 1
        super().__init__(ID)
        self.units = units
        self.activation = activation
        self.bias = bias
        self.dropout = dropout
        self.dropout_probability = dropout_probability

    def __deepcopy__(self, memodict={}):
        """ Does not retain connectivity """
        new_dense = Dense(self.ID, self.units, self.activation, self.bias, self.dropout, self.dropout_probability)
        return self.transfer_values(new_dense)

    def transfer_values(self, other):
        other.nodeID = self.nodeID
        other.ID = self.ID
        other.units = self.units
        other.activation = self.activation
        other.bias = self.bias
        other.dropout = self.dropout
        other.dropout_probability = self.dropout_probability
        return other

    def to_keras(self):
        from tensorflow import keras
        dense = keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.bias,
            name=self.ID
        )
        if self.dropout:
            return keras.layers.Dropout(rate=self.dropout_probability)(dense)
        return dense

    def find_shape(self):
        return (None, self.units)

    def set_new_id(self):
        global counter
        name = self.ID.split("_")[0].strip()
        self.ID = f"{name}_{counter}"
        counter += 1

class DenseS(Dense):
    _min_units = 10
    _max_units = 100
    _definite_units = 150

    def __init__(self, ID=None, dropout=True, dropout_probability=0.3):
        global counter
        if not ID:
            ID = "{}_{}".format("DenseS", counter)
            counter += 1
        super().__init__(
            ID=ID,
            units=self._definite_units,  # random.randint(self._min_units, self._max_units),
            dropout=dropout,
            dropout_probability=dropout_probability
        )

    def __copy__(self):
        return self.transfer_values(DenseS())

class DenseM(Dense):
    _min_units = 100
    _max_units = 500
    _definite_units = 750

    def __init__(self, ID=None, dropout=True, dropout_probability=0.3):
        global counter
        if not ID:
            ID = "{}_{}".format("DenseM", counter)
            counter += 1
        super().__init__(
            ID=ID,
            units=self._definite_units,  # random.randint(self._min_units, self._max_units),
            dropout=dropout,
            dropout_probability=dropout_probability
        )

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(DenseM())

class DenseL(Dense):
    _min_units = 1000
    _max_units = 2500
    _definite_units = 1500

    def __init__(self, ID=None, dropout=True, dropout_probability=0.3):
        global counter
        if not ID:
            ID = "{}_{}".format("DenseL", counter)
            counter += 1
        super().__init__(
            ID=ID,
            units=self._definite_units,  # random.randint(self._min_units, self._max_units),
            dropout=dropout,
            dropout_probability=dropout_probability
        )

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(DenseL())
