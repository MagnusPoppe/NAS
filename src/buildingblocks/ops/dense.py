import random
import time

from src.buildingblocks.ops.operation import Operation

counter = 0
class Dropout(Operation):

    def __init__(self):
        global counter # TODO: Skriv om til counter...
        super().__init__("Dropout_{}".format(counter))
        counter += 1
        self.rate = 0.5

    def __deepcopy__(self, memodict={}):
        """ Does not retain connectivity """
        new_dropout = Dropout()
        new_dropout.ID = self.ID
        new_dropout.rate = self.rate
        return new_dropout

    def to_keras(self):
        from tensorflow import keras
        return keras.layers.Dropout(rate=self.rate, name=self.ID)

    def find_shape(self):
        shapes = [p.find_shape() for p in self.prev]
        if len(shapes) == 1: return shapes[0]
        elif len(shapes) > 1:
            dims = shapes[0]
            for shape in shapes[1:]:
                for dim in range(len(shape)):
                    if shape[dim] != None:
                        dims[dim] += shape[dim]

class Dense(Operation):

    def __init__(self, ID, units, activation="relu", bias=True):
        global counter
        if not ID:
            ID = "{}-{}".format(ID, counter)
            counter += 1
        super().__init__(ID)
        self.units = units
        self.activation = activation
        self.bias = bias

    def __deepcopy__(self, memodict={}):
        """ Does not retain connectivity """
        new_dense = Dense(self.ID, self.units, self.activation, self.bias)
        return self.transfer_values(new_dense)

    def transfer_values(self, other):
        other.nodeID = self.nodeID
        other.ID = self.ID
        other.units = self.units
        other.activation = self.activation
        other.bias = self.bias
        return other

    def to_keras(self):
        from tensorflow import keras
        return keras.layers.Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.bias,
            name=self.ID
        )

    def find_shape(self):
        return (None, self.units)


class DenseS(Dense):
    _min_units = 10
    _max_units = 30

    def __init__(self, ID=None):
        global counter
        if not ID:
            ID = "{}-{}".format("DenseS", counter)
            counter += 1
        super().__init__(
            ID=ID,
            units=random.randint(self._min_units, self._max_units)
        )

    def __copy__(self):
        return self.transfer_values(DenseS())

class DenseM(Dense):
    _min_units = 30
    _max_units = 100

    def __init__(self, ID=None):
        global counter
        if not ID:
            ID = "{}-{}".format("DenseM", counter)
            counter += 1
        super().__init__(
            ID=ID,
            units=random.randint(self._min_units, self._max_units)
        )

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(DenseM())

class DenseL(Dense):
    _min_units = 100
    _max_units = 400

    def __init__(self, ID=None):
        global counter
        if not ID:
            ID = "{}-{}".format("DenseL", counter)
            counter += 1
        super().__init__(
            ID=ID,
            units=random.randint(self._min_units, self._max_units)
        )

    def __deepcopy__(self, memodict={}):
        return self.transfer_values(DenseL())