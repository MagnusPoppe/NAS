import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
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
    # y_test = keras.utils.to_categorical(y_test, num_classes=classes)
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
        model, device, epochs, batch_size=64, compiled=False, shuffle_interval=3
    ):
        with tf.device(device):
            # DEFINING FUNCTIONS FOR COMPILATION
            if not compiled:
                optimizer = keras.optimizers.Adam(lr=0.001)
                loss = keras.losses.categorical_crossentropy
                model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

            history = {}
            # for epoch in range(0, training_epochs, shuffle_interval):
                # SHUFFLE DATASET:
            labels, data = shuffle(x_train, y_train)

            # RUNNING TRAINING:
            metric = model.fit(
                data,
                labels,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(x_val, y_val),
            )
            for key, value in metric.history.items():
                try:
                    history[key] += value
                except KeyError:
                    history[key] = value
        return history

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


def shuffle(x_train, y_train):
    index = np.array(list(range(len(y_train))))
    np.random.shuffle(index)
    data = np.zeros(x_train.shape)
    labels = np.zeros(y_train.shape)
    for new, old in enumerate(index):
        data[new] = x_train[old]
        labels[new] = y_train[old]
    return labels, data


def main(individ, epochs, config, device):
    from src.frameworks.keras_decoder import assemble
    # Setup:

    try:
        import setproctitle
        setproctitle.setproctitle("EA-NAS-TRAINER " + device.device)
    except ImportError:
        pass
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    device_id = device.device.split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id


    training, evalutation, name, inputs = configure(config.classes_in_classifier, device)
    compiled = False
    if individ.saved_model:
        compiled = True
        with tf.device(device.device):
            model = keras.models.load_model(individ.saved_model)
    else:
        model = assemble(individ, config.input_format, config.classes_in_classifier)
        if individ.predecessor and individ.predecessor.saved_model:
            predecessor_model = keras.models.load_model(individ.predecessor.saved_model)
            transfer_model_weights(model, predecessor_model)

    training_history = training(
        model=model,
        device=device.device,
        epochs=epochs,
        batch_size=config.batch_size,
        compiled=compiled,
    )
    report = evalutation(model, device.device, compiled=True)
    return model, training_history, report


def execnet_setup(individ_str, config_str, server_str) -> tuple:
    import pickle
    individ = pickle.loads(individ_str)
    config = pickle.loads(config_str)
    server = pickle.loads(server_str)
    return individ, config, server


if __name__ == "__channelexec__":
    from firebase.upload import save_model_image

    # Reading input from main process:
    individ_str, config_str, server_str, job = channel.receive()
    individ, config, server = execnet_setup(individ_str, config_str, server_str)

    # Running training:
    try:
        model, training_history, after = main(individ, config, server)

        # Saving keras model and image of model:
        model_path = os.path.join(individ.absolute_save_path(config), "model.h5")
        image_path = os.path.join(individ.absolute_save_path(config), individ.ID + ".png")
        keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)
        save_model_image(model, image_path)

        channel.send(
            json.dumps(
                {
                    "job": job,
                    "image": image_path,
                    "model": model_path,
                    "accuracy": training_history["acc"],
                    "validation accuracy": training_history["val_acc"],
                    "loss": training_history["loss"],
                    "validation loss": training_history["val_loss"],
                    "eval": {
                        "epoch": str(len(training_history) + len(individ.fitness) - 2),
                        "accuracy": after,
                    },
                }
            )
        )
    except Exception as e:
        channel.send(str(e))

