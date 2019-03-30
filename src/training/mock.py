import random


def apply_random_increases(li):
    if not li:
        li = [.10]
    for _ in range(10):
        li += [li[-1] * 1.01] if random.uniform(0, 1) > 0.5 else [li[-1]*1.01]
    return li


def fake_report():
    def fake_numbers():
        return {
            "precision": random.uniform(0, 1),
            "recall": random.uniform(0, 1),
            "f1-score": random.uniform(0, 1),
            "support": 1000
        }
    report = {}
    for cls in list(range(10)) + ["micro avg", "macro avg", "weighted avg"]:
        report[cls] = fake_numbers()
    return report

def start(population, config):
    for individ in population:
        individ.fitness = apply_random_increases(individ.fitness)
        individ.loss = apply_random_increases(individ.loss)
        individ.validation_fitness = apply_random_increases(individ.validation_fitness )
        individ.validation_loss = apply_random_increases(individ.validation_loss)
        individ.evaluation[10] = random.uniform(0, 1)
        individ.epochs_trained += 10
        individ.report[individ.epochs_trained] = fake_report()
        individ.saved_model = None
        individ.model_image_path = None
    return population
