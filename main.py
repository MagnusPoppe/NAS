import random

from modules.convolution import Conv5x5, Conv3x3
from modules.dense import DenseS, DenseM, DenseL, Dropout
from modules.module import Module
from tensorflow import keras

operators2D = [Conv3x3, Conv5x5]
operators1D = [DenseS, DenseM, DenseL, Dropout]

def random_sample(collection):
    # Selecting a random operation and creating an instance of it.
    return collection[random.randint(0, len(collection) - 1)]


def test_model(model):
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
    model.summary()

    # RUNNING TRAINING:
    training = model.fit(
        x_train,
        keras.utils.to_categorical(y_train, num_classes=10),
        epochs=10,
        batch_size=64,
        validation_data=(
            x_val, keras.utils.to_categorical(y_val, num_classes=10))
    )

    test_metrics = model.evaluate(x_test, keras.utils.to_categorical(y_test, num_classes=10), verbose=0)
    print("\n".join(["{}: {}".format(metric, score) for metric, score in zip(model.metrics_names, test_metrics)]))


def keras_functional_api_proof_of_concept():
    input = keras.layers.Input(shape=(784,))

    # Creating main branch:
    main_layer1 = keras.layers.Dense(units=400, activation="relu")(input)
    main_layer2 = keras.layers.Dense(units=400, activation="relu")(main_layer1)
    main_layer3 = keras.layers.Dense(units=400, activation="relu")(main_layer2)
    main_layer4 = keras.layers.Dense(units=400, activation="relu")(main_layer3)

    # Creating branch:
    branch_layer1 = keras.layers.Dense(units=400, activation="relu")(main_layer2)
    branch_layer2 = keras.layers.Dense(units=400, activation="relu")(branch_layer1)
    branch_layer3 = keras.layers.Dense(units=400, activation="relu")(branch_layer2)

    # Merging:
    concat = keras.layers.concatenate([main_layer4, branch_layer3])
    output = keras.layers.Dense(units=10, activation="softmax")(concat)

    return keras.models.Model(inputs=[input], outputs=[output])


if __name__ == '__main__':
    root = Module()

    # Adding operations to graph:
    root += random_sample(operators1D)()
    root += random_sample(operators1D)()
    root += random_sample(operators1D)()
    root += random_sample(operators1D)()

    # Adding an alternative route to the graph
    prev = root.children[2]  # Second last
    end = root.children[3]   # last
    for i in range(3):
        op = random_sample(operators1D)()
        prev.next += [op]
        op.prev += [prev]
        root.children += [op]
        prev = op
    op.next += [end]
    end.prev += [op]


    root.insert(root.children[5], root.children[2], random_sample(operators1D)())
    root.insert(root.children[0], root.children[-1], random_sample(operators1D)())

    print(root)
    root.visualize()
    model = root.compile(input_shape=(784,), classes=10)
    keras.utils.plot_model(model, to_file='model.png')
    test_model(model)
