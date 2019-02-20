import json
import unittest
import os

from src.frameworks.keras_decoder import assemble

os.chdir("..")
from tensorflow import keras
import tensorflow as tf

from datasets.cifar10 import main, shuffle
from firebase.upload import save_model_image


class TestTraining(unittest.TestCase):

    def test_load_model_correctly_from_file(self):
        import pickle, os
        from src.frameworks.keras_decoder import assemble

        # Removing all debugging output from TF:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        # Reading input fixtures:
        picklepath = "./tests/fixtures/Didrik/v0/genotype.obj"
        for i in range(10):
            with open(picklepath, "rb") as f:
                individ = pickle.load(f)
            with open("./datasets/cifar10-local.json", "r") as f:
                config = json.load(f)
                server = config['servers'][0]

            # Running training:
            model, training_history, after = main(individ, config, server)

            # Saving keras model and image of model:
            os.makedirs("./tests/fixtures/Didrik/v0/testing/", exist_ok=True)
            model_path = "./tests/fixtures/Didrik/v0/testing/model.h5"
            image_path = "./tests/fixtures/Didrik/v0/testing/" + individ.ID + ".png"
            keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)
            save_model_image(model, image_path)

            individ.saved_model = model_path
            individ.model_image_path = image_path

            print('After evaluation', after)
            picklepath = "./tests/fixtures/Didrik/v0/testing/genotype.obj"
            individ.clean()
            with open(picklepath, "wb") as f:
                 pickle.dump(individ, f)


    def test_training_integration(self):
        import pickle, os
        os.chdir("..")
        from src.buildingblocks.module import Module
        from src.evolutionary_operations import mutation_operators as op

        from src.buildingblocks.ops.convolution import Conv3x3, Conv5x5
        from src.buildingblocks.ops.dense import (
            DenseS as DenseSmall,
            DenseM as DenseMedium,
            DenseL as DenseLarge,
            Dropout,
        )
        from src.buildingblocks.ops.pooling import MaxPooling2x2, AvgPooling2x2

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        module = Module()
        module = op.append(module, Conv3x3())
        module = op.append(module, Dropout())
        module = op.append(module, Conv3x3())
        module = op.append(module, Dropout())
        module = op.append(module, Conv3x3())
        module = op.append(module, Dropout())
        Module = op.append(module, MaxPooling2x2())
        module = op.append(module, DenseLarge())
        module = op.append(module, Dropout())
        module = op.append(module, DenseLarge())

        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_val = x_train[45000:] / 255
        y_val = y_train[45000:]
        x_train = x_train[:45000] / 255
        y_train = y_train[:45000]
        x_test = x_test / 255

        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        y_val = keras.utils.to_categorical(y_val, num_classes=10)

        labels, data = shuffle(x_train, y_train)
        with tf.device("/GPU:0"):
            keras.backend.set_session(
                tf.Session(
                    config=tf.ConfigProto(
                        gpu_options=tf.GPUOptions(
                            allow_growth=False,
                            per_process_gpu_memory_fraction=1.0,
                        ),
                        allow_soft_placement=True,
                        log_device_placement=True
                    )
                )
            )
            model = assemble(module, in_shape=(32,32,3), classes=10)
            metrics = model.fit(
                data,
                labels,
                epochs=10,
                batch_size=250,
                verbose=0,
                validation_data=(x_val, y_val),
            )

            results = model.evaluate(x_test, y_test)







