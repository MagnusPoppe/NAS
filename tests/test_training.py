import json
import os
os.chdir("..")
import unittest
from tensorflow import keras

from datasets.cifar10 import main
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
            with open("./datasets/cifar10-home-ssh.json", "r") as f:
                config = json.load(f)
                server = config['servers'][0]

            # Running training:
            model, training_history, before, after = main(individ, config, server)

            # Saving keras model and image of model:
            os.makedirs("./tests/fixtures/Didrik/v0/testing/", exist_ok=True)
            model_path = "./tests/fixtures/Didrik/v0/testing/model.h5"
            image_path = "./tests/fixtures/Didrik/v0/testing/" + individ.ID + ".png"
            keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)
            save_model_image(model, image_path)

            individ.saved_model = model_path
            individ.model_image_path = image_path

            print({'before eval': before, 'after eval': after})
            picklepath = "./tests/fixtures/Didrik/v0/testing/genotype.obj"
            individ.clean()
            with open(picklepath, "wb") as f:
                 pickle.dump(individ, f)