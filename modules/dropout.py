from modules.dense import Dense
from modules.operation import Operation
from tensorflow import keras

class Dropout(Operation):

    def __init__(self):
        super().__init__("Dropout", [Dense])
        self.rate = 0.5

    def to_keras(self):
        return keras.layers.Dropout(rate=self.rate)