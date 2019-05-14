import copy

from src.MOOA.NSGA_II import weighted_overfit_score, nsga_ii
from src.buildingblocks.module import Module
from src.configuration import Configuration
from src.output import generation_finished
import src.jobs.jobs as workers


def try_finish(population: [Module], config: Configuration, moo_ops: callable) -> [Module]:
    print(f"--> Possible final solution discovered. Checking...")

    # Changing settings of training steps:
    original_training_settings = copy.deepcopy(config.training)
    config.training.use_restart = False
    config.training.fixed_epochs = True
    config.training.epochs = 100

    # Finding the best networks:
    best = population[:config.compute_capacity(maximum=False)]

    # Performing training step:
    best = workers.start(best, config)

    # Reset settings and return:
    config.training = original_training_settings

    best.sort(key=weighted_overfit_score(config), reverse=True)
    if any(ind.test_acc() >= config.training.acceptable_scores for ind in best):
        generation_finished(best, config, "--> Found final solution:")
        config.results.store_generation(best, config.generation + 1)
        return best, True
    else:
        # A final solution was not found... Keep the best individs:
        population = best + population
        population = nsga_ii(
            population,
            moo_ops.classification_objectives(config),
            moo_ops.classification_domination_operator(
                moo_ops.classification_objectives(config)
            ),
            config
        )
        keep = len(population) - config.population_size
        population, removed = population[keep:], population[:keep]
        generation_finished(population, config, "--> Leaderboards after final solution try failed:")
        generation_finished(removed, config, "--> Removed after final solution try failed:")
        return population, False
