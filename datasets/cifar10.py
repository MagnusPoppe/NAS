import tensorflow as tf
from tensorflow import keras

from firebase.upload import upload_image
from frameworks.weight_transfer import transfer_predecessor_weights


def configure(classes, server):  # -> (function, function):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_val = x_train[45000:] / 255
    y_val = y_train[45000:]
    x_train = x_train[:45000] / 255
    y_train = y_train[:45000]
    x_test = x_test / 255

    y_train = keras.utils.to_categorical(y_train, num_classes=classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=classes)


    keras.backend.set_session(tf.Session(config=tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=server['allow gpu memory growth'],
            per_process_gpu_memory_fraction = server['memory per process']
        )
    )))


    def train(model, device, epochs=1.2, batch_size=64):
        training_epochs = int(epochs * len(model.layers)) if epochs > 0 else 1
        with tf.device(device):
            # DEFINING FUNCTIONS FOR COMPILATION
            sgd = keras.optimizers.Adam(lr=0.01)
            loss = keras.losses.categorical_crossentropy

            # RUNNING TRAINING:
            model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
            metrics = model.fit(
                x_train,
                y_train,
                epochs=training_epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(x_val, y_val)
            )
            return metrics.history['val_acc'][-1]

    def evaluate(population: list, device:str, compiled=True, prefix="--> "):
        print(prefix + "Evaluating {} models".format(len(population)))
        for individ in population:
            with tf.device(device):
                # DEFINING FUNCTIONS FOR COMPILATION
                sgd = keras.optimizers.Adam(lr=0.01)
                loss = keras.losses.categorical_crossentropy

                model = individ.keras_operation
                if not compiled:
                    model.compile(loss=loss, optimizer=sgd, metrics=['accuracy'])
                metrics = model.evaluate(x_test, y_test, verbose=0)
                individ.fitness = metrics[1]  # Accuracy


    return train, evaluate, "CIFAR 10", (32, 32, 3)

if __name__ == '__main__':
    import pickle, os, json
    from frameworks.keras_decoder import assemble
    with open("results/test-data/Felicia/v2/genotype.obj", "rb") as f:
        individ = pickle.load(f)
    with open("datasets/cifar10-config.json", "r") as f:
        config = json.load(f)

    model = assemble(individ, config['input'], config['classes'])
    predecessor_model = keras.models.load_model("results/test-data/Felicia/v0/model.h5")
    transfer_predecessor_weights(model, predecessor_model)


    training, evalutation, name, inputs = configure(config['classes'])
    print(training(model, config['device'], config['epochs'], config['batch size']))

if __name__ == '__channelexec__':
    import pickle, os
    from frameworks.keras_decoder import assemble

    # Removing all debugging output from TF:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    individ_str, config_str, server_id = channel.receive()
    individ = pickle.loads(individ_str)
    config = pickle.loads(config_str)
    classes = 10

    training, evalutation, name, inputs = configure(config['classes'], config['servers'][server_id])
    if individ.saved_model:
        model = keras.models.load_model(individ.saved_model)
    else:
        model = assemble(individ, config['input'], config['classes'])
        if individ.predecessor:
            predecessor_model = keras.models.load_model(individ.predecessor.saved_model)
            transfer_predecessor_weights(model, predecessor_model)

    fitness = training(
        model=model,
        device=config['servers'][server_id]['device'],
        epochs=config['epochs'],
        batch_size=config['batch size']
    )

    # Saving keras model:
    model_path = individ.get_absolute_module_save_path(config)
    model_path = os.path.join(model_path, "model.h5")
    model.save(model_path)


    image_path = upload_image(individ, model, config['run id'])

    channel.send((float(fitness), model_path, image_path))