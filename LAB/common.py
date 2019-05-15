from src.buildingblocks.module import Module
from src.configuration import Configuration


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


def find_filetype_recursivly(directory: str, filetype: str):
    import os
    files_found = []
    for content in os.listdir(directory):
        path = os.path.join(directory, content)
        if os.path.isfile(path) and filetype in content:
            files_found += [path]
        elif os.path.isdir(path):
            files_found += find_filetype_recursivly(path, filetype)
    return files_found


class Simulation:
    def __init__(self, config, population, generations):
        self.config = config  # type: Configuration
        self.population = population  # type: [Module]
        self.generations = generations  # type: [[Module]]
        self.avg_acc = []
        self.avg_val_acc = []
        self.avg_test_acc = []
        self.max_acc = []
        self.max_val_acc = []
        self.max_test_acc = []
        self.min_acc = []
        self.min_test_acc = []
        self.min_val_acc = []
        for generation in self.generations:
            self.avg_acc += [sum([x.acc() for x in generation]) / len(generation)]
            self.avg_val_acc += [sum([x.val_acc() for x in generation]) / len(generation)]
            self.avg_test_acc += [sum([x.test_acc() for x in generation]) / len(generation)]
            self.max_acc += [max([x.acc() for x in generation])]
            self.max_val_acc += [max([x.val_acc() for x in generation])]
            self.max_test_acc += [max([x.test_acc() for x in generation])]
            self.min_acc += [min([x.acc() for x in generation])]
            self.min_val_acc += [min([x.val_acc() for x in generation])]
            self.min_test_acc += [min([x.test_acc() for x in generation])]

    @staticmethod
    def read_all_genotypes_per_generation(config_dir: str, results_dir: str, experiment_name: str):
        import os, pickle
        print(f"Loading {experiment_name}")
        config = Configuration.from_json(os.path.join(config_dir, f"{experiment_name}.json"))
        experiment_results = os.path.join(results_dir, experiment_name)
        # Read all genotypes first
        population = []

        files = find_filetype_recursivly(experiment_results, filetype=".obj")
        for file_path in files:
            with open(file_path, "rb") as ptr:
                population += [pickle.load(ptr)]
                print(f" - Reading files, {len(population) / len(files) * 100:.0f}% complete" + "\t" * 3, end="\r")
        print()
        # Map genotype to generation
        generations = []
        directory = os.path.join(experiment_results, "generations")
        gens = [int(generation) for generation in os.listdir(directory)]
        gens.sort()
        for gen in gens:
            gen_dir = os.path.join(directory, str(gen))
            generation = []
            for identifier in os.listdir(gen_dir):
                identifier = identifier.strip("'")
                for ind in population:
                    if ind.ID == identifier:
                        generation += [ind]
                        break
            generations += [generation]
        print(" - population sorted into generations")
        return Simulation(config, population, generations)


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
