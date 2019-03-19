import pandas as pd
from src.buildingblocks.module import Module
from src.configuration import Configuration


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

def _save_kears_model_images(args: str):
    from tensorflow import keras
    import pickle, os
    individ, config = pickle.loads(args)
    path = individ.absolute_save_path(config)
    model = keras.models.load_model(os.path.join(path, "model.h5"))
    keras.utils.plot_model(model, to_file=os.path.join(path, "img.png"))
    return os.path.join(path, "img.png")

def create_images(individs: [Module], config):
    import multiprocessing as mp
    import pickle
    args = [pickle.dumps((individ, config)) for individ in individs]
    pool = mp.Pool()
    mapper = pool.map_async(_save_kears_model_images, args)
    pool.close()
    results = mapper.get()
    for i in range(len(individs)):
        individs[i].model_image_path = results[i]
    return individs

if __name__ == '__main__':
    import os
    # Usually like: ./results/<run-name>/<individ-name>/<version>/
    module_stores = [
        "./results/simulation01/anne-sofie/v0/",
        "./results/simulation01/anne-sofie/v2/",
        "./results/simulation01/anne-sofie/v4/"
    ]

    # Configuration file:
    config_file = "./datasets/tester.json"
    config = Configuration.from_json(config_file)
    modules = []
    for model_path in module_stores:
        module = load_module(os.path.join(model_path, "genotype.obj"))
        print("Loaded module {}".format(module.ID))
        modules += [module]
    # progress_report(modules[0].report)
    for x in create_images(modules, config):
        print(f"Saved image to {x.model_image_path}")
