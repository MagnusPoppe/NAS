import unittest

from tensorflow import keras

from main import random_sample, operators1D
from modules.dense import DenseS
from modules.module import Module


class TestModuleCompile(unittest.TestCase):

    def test_A_module_compiles_sequential_model(self):
        root = Module()
        root.ID = "Root"

        # Adding operations to graph:
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()

        model = root.compile(input_shape=(784,), classes=10)
        self.assertTrue(isinstance(model, keras.models.Model), "Does not compile sequential model.")

    def test_B_module_compiles_with_branches(self):
        root = Module()
        root.ID = "Root"

        # Adding operations to graph:
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()

        branch_start = root.children[0]
        branch_op1 = random_sample(operators1D)()
        branch_op2 = random_sample(operators1D)()
        branch_end = root.children[1]

        root.insert(branch_start, branch_end, branch_op1)
        root.insert(branch_op1, branch_end, branch_op2)

        model = root.compile(input_shape=(784,), classes=10)
        self.assertTrue(isinstance(model, keras.models.Model), "Does not compile model with branches.")

    def test_C_module_compiles_with_single_sub_module(self):
        root = Module()
        root.ID = "Root"

        # Adding operations to graph:
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()
        root += Module(ID = "sub-module1").append(DenseS()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)())

        branch_start = root.children[0]
        branch_op1 = random_sample(operators1D)()
        branch_op2 = random_sample(operators1D)()
        branch_end = root.children[1] # This is a module.

        root.insert(branch_start, branch_end, branch_op1)
        root.insert(branch_op1, branch_end, branch_op2)

        model = root.compile(input_shape=(784,), classes=10)
        self.assertTrue(isinstance(model, keras.models.Model), "Does not compile model with branches.")

    def test_D_module_compiles_with_only_sub_modules(self):

        # Testing submodule operator
        sub_module1 = Module().append(DenseS()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)())
        sub_module1.ID = "sub-module1"

        sub_module2 = Module().append(DenseS()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)())
        sub_module2.ID = "sub-module2"

        root = Module().append(sub_module1).append(sub_module2)
        model = root.compile(input_shape=(784,), classes=10)
        run_keras_model(model)

    def test_E_complex_module_works_with_keras(self):
        root = Module()
        root.ID = "Root"

        # Adding operations to graph:
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()
        root += random_sample(operators1D)()

        # Adding an alternative route to the graph
        prev = root.children[2]  # Second last
        end = root.children[3]  # last
        for i in range(3):
            op = random_sample(operators1D)()
            prev.next += [op]
            op.prev += [prev]
            root.children += [op]
            prev = op
        op.next += [end]
        end.prev += [op]

        # Testing insert operator
        root.insert(root.children[5], root.children[2], random_sample(operators1D)())
        root.insert(root.children[0], root.children[3], random_sample(operators1D)())

        # Testing submodule operator
        sub_module = Module().append(DenseS()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)()) \
            .append(random_sample(operators1D)())
        sub_module.ID = "sub-module"

        root.insert(root.children[1], root.children[2], sub_module)

        model = root.compile(input_shape=(784,), classes=10)
        keras.utils.plot_model(model, to_file='tested_model.png')

        try:
            run_keras_model(model)
        except Exception as e:
            self.fail(msg=e)


def run_keras_model(model):
    def fix(data):
        import numpy as np
        return np.reshape(data, (len(data), 784))

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # VALIDATION DATA:
    x_val = fix(x_train[50000:])
    y_val = y_train[50000:]
    x_val = x_val.astype('float32')
    x_val /= 255

    # TRAINING DATA:
    x_train = fix(x_train[:50000])
    y_train = y_train[:50000]
    x_train = x_train.astype('float32')
    x_train /= 255

    # TEST DATA:
    x_test = fix(x_test)
    x_test = x_test.astype('float32')
    x_test /= 255

    # for _ in range(10):

    # DEFINING FUNCTIONS AND COMPILING
    sgd = keras.optimizers.Adam(lr=0.01)
    loss = keras.losses.categorical_crossentropy
    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])

    # RUNNING TRAINING:
    training = model.fit(
        x_train,
        keras.utils.to_categorical(y_train, num_classes=10),
        epochs=2,
        batch_size=64,
        verbose=0,
        validation_data=(
            x_val, keras.utils.to_categorical(y_val, num_classes=10))
    )

    test_metrics = model.evaluate(x_test, keras.utils.to_categorical(y_test, num_classes=10), verbose=0)
    print( "\n".join(["{}: {}".format(metric, score) for metric, score in zip(model.metrics_names, test_metrics)]))


if __name__ == '__main__':
    unittest.main()