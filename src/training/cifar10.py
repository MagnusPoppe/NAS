import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from sklearn.metrics import classification_report

from src.frameworks.weight_transfer import transfer_model_weights


def configure(classes, device) -> (callable, callable):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_val = x_train[45000:] / 255
    y_val = y_train[45000:]
    x_train = x_train[:45000] / 255
    y_train = y_train[:45000]
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, num_classes=classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=classes)

    with tf.device(device.device):
        keras.backend.set_session(
            tf.Session(
                config=tf.ConfigProto(
                    gpu_options=tf.GPUOptions(
                        allow_growth=device.allow_memory_growth,
                        per_process_gpu_memory_fraction=device.memory_per_process,
                    ),
                    allow_soft_placement=True
                )
            )
        )

    def train(
        model, device, epochs, batch_size=64, compiled=False
    ):
        with tf.device(device):
            # DEFINING FUNCTIONS FOR COMPILATION
            if not compiled:
                optimizer = keras.optimizers.Adam(lr=0.001)
                loss = keras.losses.categorical_crossentropy
                model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

            # RUNNING TRAINING:
            metric = model.fit(
                x=x_train,
                y=y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                shuffle=True,
                validation_data=(x_val, y_val),
            )
        return metric.history

    def evaluate(model, device: str, compiled=True):
        with tf.device(device):
            if not compiled:
                # DEFINING FUNCTIONS FOR COMPILATION
                sgd = keras.optimizers.Adam(lr=0.001)
                loss = keras.losses.categorical_crossentropy
                model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])

            # EVALUATING
            predictions = model.predict(x_test)
            pred = np.argmax(predictions, axis=1)
        return classification_report(y_test, pred, output_dict=True)
    return train, evaluate, "CIFAR 10", (32, 32, 3)


def prepare_model(config, device, individ):
    from src.frameworks.keras import module_to_model as assemble
    import os
    compiled = False
    if individ.saved_model and os.path.isfile(individ.saved_model):
        with tf.device(device.device):
            model = keras.models.load_model(individ.saved_model)
        compiled = True
    else:
        model = assemble(individ, config.input_format, config.classes_in_classifier)
        if individ.predecessor and individ.predecessor.saved_model:
            predecessor_model = keras.models.load_model(individ.predecessor.saved_model)
            transfer_model_weights(model, predecessor_model)
    return compiled, model


def prepare_pattern_model(config, device, individ):
    from src.frameworks.keras import module_to_model as assemble
    model = assemble(individ, config.input_format, config.classes_in_classifier)
    # if individ.predecessor and individ.predecessor.saved_model:
    #     predecessor_model = keras.models.load_model(individ.predecessor.saved_model)
    #     transfer_model_weights(model, predecessor_model)
    return False, model


def main(individ, epochs, config, device):
    training, evalutation, name, inputs = configure(config.classes_in_classifier, device)

    preparation = prepare_pattern_model if config.type == "PatternNets" else prepare_model
    compiled, model = preparation(config, device, individ)
    training_history = training(
        model=model,
        device=device.device,
        epochs=epochs,
        batch_size=config.batch_size,
        compiled=compiled,
    )
    report = evalutation(model, device.device, compiled=True)
    return model, training_history, report
