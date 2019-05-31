from src.buildingblocks.module import Module
from src.buildingblocks.pattern import Pattern
from src.configuration import Configuration


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

def isint(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


class Simulation:
    def __init__(self, config, population, generations, patterns=None, pattern_generations=None):
        self.config = config  # type: Configuration
        self.population = population  # type: [Module]
        self.generations = generations  # type: [[Module]]
        self.patterns = patterns
        self.pattern_generations = pattern_generations
        self.set_acc_metrics()
        
    def set_acc_metrics(self):
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
            self.avg_val_acc += [
                sum([x.val_acc() for x in generation]) / len(generation)
            ]
            self.avg_test_acc += [
                sum([x.test_acc() for x in generation]) / len(generation)
            ]
            self.max_acc += [max([x.acc() for x in generation])]
            self.max_val_acc += [max([x.val_acc() for x in generation])]
            self.max_test_acc += [max([x.test_acc() for x in generation])]
            self.min_acc += [min([x.acc() for x in generation])]
            self.min_val_acc += [min([x.val_acc() for x in generation])]
            self.min_test_acc += [min([x.test_acc() for x in generation])]

    def alltime_best(self):
        best = None
        generation = 0
        for i, gen in enumerate(self.generations):
            for ind in gen:
                if not best:
                    best = ind
                    continue
                if best.test_acc() < ind.test_acc():
                    best = ind
                    generation = i
        return best, generation

    @staticmethod
    def read_all_genotypes_per_generation(
            config_dir: str, results_dir: str, experiment_name: str
    ):
        import os, pickle

        print(f"Loading {experiment_name}")
        config = Configuration.from_json(
            os.path.join(config_dir, f"{experiment_name}.json"), validation=False
        )
        experiment_results = os.path.join(results_dir, experiment_name)
        # Read all genotypes first
        population = []
        patterns = []

        files = find_filetype_recursivly(experiment_results, filetype=".obj")
        for file_path in files:
            with open(file_path, "rb") as ptr:
                obj = pickle.load(ptr)
                if not isinstance(obj, Pattern):
                    population += [obj]
                else:
                    patterns += [obj]
                print(
                    f" - Reading files, {(len(population) + len(patterns)) / len(files) * 100:.0f}% complete"
                    + "\t" * 3,
                    end="\r",
                )
        print()
        # Map genotype to generation
        generations = []
        pattern_generations = []
        directory = os.path.join(experiment_results, "generations")
        gens = [int(generation) for generation in os.listdir(directory) if isint(generation)]
        gens.sort()
        for gen in gens:
            gen_dir = os.path.join(directory, str(gen))
            generation = []
            p_generation = []
            for identifier in os.listdir(gen_dir):
                identifier = identifier.strip("'")
                for ind in population:
                    if ind.ID == identifier:
                        generation += [ind]
                        break
                else:
                    for ind in patterns:
                        if ind.ID == identifier:
                            p_generation += [ind]
                            break
            generations += [generation]
            pattern_generations += [p_generation]
        print(" - population sorted into generations")
        return Simulation(config, population, generations, patterns, pattern_generations)
