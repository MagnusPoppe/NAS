import random

from src.pattern_nets import evaluation as pattern_evaluation
from src.training.prepare_training import apply_ea_nas_results

def apply_random_increases(li):
    if not li:
        li = [.10]
    for _ in range(10):
        li += [li[-1] * 1.01] if random.uniform(0, 1) > 0.5 else [li[-1]*1.01]
    return li


def fake_report():
    import numpy as np
    report = {}
    classes = list(range(10)) + ["micro avg", "macro avg", "weighted avg"]
    numbers = np.random.random(len(classes) * 3)
    i = 0
    for cls in classes:
        report[str(cls)] = {
            "precision": numbers[i],
            "recall": numbers[i+1],
            "f1-score": numbers[i+2],
            "support": 1000
        }
        i += 3
    return report


def start(population, config):
    for i, individ in enumerate(population):
        report = fake_report()
        result = {
            "job": i,
            "epochs": 10,
            "accuracy": apply_random_increases(individ.fitness),
            "validation accuracy": apply_random_increases(individ.validation_fitness),
            "test accuracy": report['weighted avg']['precision'],
            "loss": apply_random_increases(individ.loss),
            "validation loss": apply_random_increases(individ.validation_loss),
            "report": report
        }

        pattern_evaluation.apply_result(individ, result, config.training.learning_rate)
        apply_ea_nas_results(individ, result, config.training.learning_rate)

    return population
