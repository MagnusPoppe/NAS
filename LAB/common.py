from src.buildingblocks.module import Module
from src.buildingblocks.pattern import Pattern
from src.configuration import Configuration
from LAB.simulation import Simulation
import os


def load_module(filepath: str) -> Module:
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


def progress_report(report: dict, name: str):
    import pandas as pd

    cifar_keys = [
        "Airplane",
        "Automobile",
        "Bird",
        "Cat",
        "Deer",
        "Dog",
        "Frog",
        "Horse",
        "Ship",
        "Truck",
    ]
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

    individ, path = pickle.loads(args)
    model = keras.models.load_model(os.path.join(path, "model.h5"))
    keras.utils.plot_model(model, to_file=os.path.join(path, "img.png"))
    return os.path.join(path, "img.png")


def create_images(individs: [Module], config=None, paths=None):
    import multiprocessing as mp
    import pickle
    if not paths:
        paths = [individ.absolute_save_path(config) for individ in individs]
    args = [pickle.dumps((individ, path)) for individ, path in zip(individs, paths)]
    pool = mp.Pool()
    mapper = pool.map_async(_save_kears_model_images, args)
    pool.close()
    results = mapper.get()
    for i in range(len(individs)):
        if individs[i]:
            individs[i].model_image_path = results[i]
    return individs

def model_params(args):
    name, model_path = args
    from tensorflow import keras

    print(f"Reading model {name}")
    model = keras.models.load_model(model_path)
    print(f"Model {name} contains parameters:", model.count_params())


if __name__ == "__main__":

    storage_dir = r"/Users/magnus/Desktop/Results"
    config_dir = f"{storage_dir}/configs"
    results_dir = f"{storage_dir}/results"

    models = [
        ("Baseline", f"{results_dir}/exp01/individs/Nikoline/454/model.h5"),
        ("Classifier Task MOO", f"{results_dir}/exp02/individs/Marcin/572/model.h5"),
        ("Architectural MOO", f"{results_dir}/exp03/individs/Lisbet/366/model.h5"),
        # ("Patterns", f"{results_dir}/exp04/individs/Nikoline/454/model.h5"),
        ("Local", f"{results_dir}/exp05/individs/Magdalena/218/model.h5"),
        ("Baseline w/o TL", f"{results_dir}/exp01-no-tl/individs/Berit/894/model.h5"),
    ]
    # paths = [os.path.dirname(model) for _, model in models]
    # create_images([None]*len(models), paths=paths)
    # import multiprocessing as mp
#
    # pool = mp.Pool()
    # mapper = pool.map_async(model_params, models)
    # pool.close()
    # results = mapper.get()

    exp06 = Simulation.read_all_genotypes_per_generation(
       config_dir, results_dir, "exp01-no-tl"
    )

