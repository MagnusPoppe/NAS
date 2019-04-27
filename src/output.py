import io
import sys

from firebase.upload import upload_population
from src.configuration import Configuration

def matrix_print(matrix):
    # Finding width of columns:
    colwise_max = []
    for x in range(len(matrix[0])):
        maximum = 0
        for y in range(len(matrix)):
            current = len(str(matrix[y][x]))
            if current > maximum:
                maximum = current
        colwise_max += [maximum + 3]

    # Building matrix string:
    string = ""
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            spaces = " " * (colwise_max[x] - len(str(cell)))
            val = str(cell)
            string += val + spaces
        string += "\n"
    return string

def get_improvement(individ):
    if len(individ.report) > 1:
        keys = list(individ.report.keys())
        impr = individ.report[keys[-1]]['weighted avg']["f1-score"] \
               - individ.report[keys[-2]]['weighted avg']["f1-score"]
        return str(round(impr, 1))
    else:
        return "-"


def col(val: str, cols: int):
    return val + " " * (cols - len(val))


def generation_finished(population, config, prefix):
    print(prefix)
    matrix = [
        ["SPECIMIN", "ACC", "VACC", "IMPR", "LR", "1", "2", "3", "4", "5", "6", "7", "8", "9", "MIAVG", "MAAVG", "WAVG"]
    ]

    if config.type == "ea-nas":
        for individ in population:
            matrix += [
                [
                    round(individ.fitness[-1] * 100, 1),
                    round(individ.validation_fitness[-1] * 100, 1),
                    get_improvement(individ),
                ] + [
                    round(report["f1-score"] * 100, 1)
                    for report in individ.report[individ.epochs_trained].values()
                ]
            ]
    else:
        for individ in population:
            result = individ.optimal_result()
            matrix += [
                [
                    round(result.accuracy[-1] * 100, 1),
                    round(result.val_accuracy[-1] * 100, 1),
                    "-",
                    result.learning_rate
                ] + [
                    round(report["f1-score"] * 100, 1)
                    for report in result.report.values()
                ]
            ]
    print(matrix_print(matrix))
    upload_population(population)

def print_config_stats(config: Configuration):
    import os
    sort_type = 'Multi-objective' if config.population_size > 15 else 'Weighted scoring'
    epochs_fixed = "(Fixed)" if config.training.fixed_epochs else "(Multiplied by network size)"
    storage_area = f"{config.results.location}/results/{config.results.name}" \
                 if config.results.location \
                 else f"{os.getcwd()}/results/{config.results.name}"

    print(f"\nConfiguration for {config.dataset_name}:")
    print(f"Evolutionary algorithm parameters:")
    print(f"\tType:                          {config.type}")
    print(f"\tPopulation size:               {config.population_size}")
    print(f"\tGenerations:                   {config.generations}")
    print(f"\tNumber of pattern/layers used: {config.min_size} - {config.max_size}")
    print(f"\tElitism sorting type:          {sort_type}")
    print(f"Neural network training:")
    print(f"\tEpochs:                        {config.training.epochs} {epochs_fixed}")
    print(f"\tMinibatch size:                {config.training.batch_size}")
    print(f"\tUse restarting:                {config.training.use_restart}")
    print(f"Servers:")
    print(f"\tNumber of servers:             {len(config.servers)}")
    print(f"\tNumber of compute devices:     {sum(len(server.devices) for server in config.servers)}")
    print(f"\tResults save location:         {storage_area}")
    print(f"\tDelete unused results:         {not config.results.keep_all}")
    print()

def print_population(population):
    print("Current population:")
    for p in population:
        print("\t-", str(p))


def no_stdout(fn):
    def wrapper(*args, **kwargs):
        # Replacing stdout temporarily:
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Running function:
        output = fn(*args, **kwargs)

        # Restoring stdout
        sys.stdout = original_stdout
        return output

    return wrapper
