import pandas as pd
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

def load_all_modules_from_run(run_filepath: str) -> [Module]:
    import os
    modules = {}
    for name in os.listdir(run_filepath):
        modules[name] = []
        for version in os.listdir(os.path.join(run_filepath, name)):
            path = os.path.join(run_filepath, name, version, "genotype.obj")
            modules[name] += [load_module(path)]
    return modules

def progress_report(report: dict, name:str) -> pd.DataFrame:
    cifar_keys = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    dataframes = [pd.DataFrame.from_dict(rep) for _, rep in report.items()]
    df = pd.concat(dataframes, keys=report.keys())
    df.name = name
    cols = []
    for col in df.columns:
        try:
            i = int(col)
            cols += [f"{i}, {cifar_keys[i]}"]
        except ValueError:
            cols += [col]
    df.columns = cols
    return df

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
    import os
    # Usually like: ./results/<run-name>/<individ-name>/<version>/
    module_stores = [
        "./results/8x075x250/Mai/v0/",
        "./results/8x075x250/Mai/v2/",
        "./results/8x075x250/Mai/v4/",
        "./results/8x075x250/Mai/v6/"
    ]

    # Configuration file:
    config_file = "./datasets/cifar-10-mp.json"
    config = load_json_config(config_file)
    modules = []
    for model_path in module_stores:
        module = load_module(os.path.join(model_path, "genotype.obj"))
        print("Loaded module {}".format(module.ID))
        modules += [module]
    progress_report(modules[0].report)
