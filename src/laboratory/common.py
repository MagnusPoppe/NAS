import pandas as pd
import os
if not os.path.basename(os.getcwd()) in ["ea-nas", "EA-architecture-search"]:
    os.chdir("../../")
from src.buildingblocks.module import Module

def load_json_config(filepath:str) -> dict:
    import json
    with open(filepath, "r") as f:
        config = json.load(f)
    return config


def load_module(filepath:str) -> Module:
    import pickle
    with open(filepath, "rb") as f:
        module = pickle.load(f)
    return module


def progress_report(report: dict) -> pd.DataFrame:
    from copy import deepcopy
    cifar_keys = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    indexes = [
        list(report.keys()),
        list(report[list(report.keys())[0]].keys())
    ]
    codes = [
        list(range(len(indexes[0]))) * 13,
        list(range(len(indexes[1]))) * 2,
    ]

    superindex = pd.MultiIndex(levels=indexes, codes=codes, names=["Epoch recorded", "Classes"])
    final_report = pd.DataFrame(columns=superindex)

    for epoch, trained_report in report.items():
        new_report = {}
        for i, metrics in enumerate(trained_report.values()):
            for metric, val in metrics:
                final_report(axis=1)[(epoch, metrics), metric] = val

    return final_report

def create_images(individs: [Module], config):
    import multiprocessing as mp

    def save_kears_model_images(individ: Module):
        from firebase.upload import save_model_image
        from tensorflow import keras
        path = individ.absolute_save_path(config)
        model = keras.models.load_model(os.path.join(path, "model.h5"))
        image_path = os.path.join(path, "img.png")
        save_model_image(model, image_path)
        return image_path

    args = [(individ) for individ in individs]
    pool = mp.Pool()
    mapper = pool.apply_async(save_kears_model_images, args=args)
    pool.close()
    results = mapper.get()
    for i in range(len(individs)):
        individs[i].model_image_path = results[i]
    return individs

if __name__ == '__main__':
    import sys
    import src.laboratory.common as fn

    # Usually like: ./results/<run-name>/<individ-name>/<version>/
    module_stores = [
        "./results/8x2/Angelika/v0/",
        "./results/8x2/Angelika/v2/"
    ]

    # Configuration file:
    config_file = "./datasets/cifar-10-mp.json"
    config = fn.load_json_config(config_file)
    modules = []
    for model_path in module_stores:
        module = fn.load_module(os.path.join(model_path, "genotype.obj"))
        print("Loaded module {}".format(module.ID))
        modules += [module]
    fn.progress_report(modules[0].report)