import copy
import sys

from src.MOOA.NSGA_II import nsga_ii, weighted_overfit_score
from src.configuration import Configuration
from src.output import generation_finished, print_population
from src.pattern_nets.initialization import initialize_patterns
from src.pattern_nets import recombine, evaluation, selection, mutator, crossover
from src.pattern_nets import moo_objectives as moo

if len(sys.argv) == 3 and sys.argv[2] == "mock":
    import src.training.mock as workers
else:
    import src.jobs.job_initializer as workers


def initialize_population(config, compute_capacity):
    # 1. Initialize population:
    if config.results.load:
        patterns = config.results.transfer_and_load_population()
    else:
        patterns = initialize_patterns(config.population_size)

    print_population(patterns)

    # 2. Evaluation of initial population. Fitness calculation
    nets = recombine.combine(patterns, compute_capacity, config.min_size, config.max_size)
    nets = workers.start(nets, config)
    patterns = evaluation.inherit_results(patterns, nets)

    generation_finished(patterns, config, f"--> Initialization Complete:")
    return patterns, nets


def main(config: Configuration):
    # 0.1 How many nets can be trained for each generation?
    solved = False
    compute_capacity = config.compute_capacity()

    # 0.2 Initializing multi objective optimisation sorting:
    moo_objectives = moo.classification_objectives(config)
    domination_operator = moo.classification_domination_operator(
        moo.classification_objectives(config)
    )

    patterns, nets = initialize_population(config, compute_capacity)

    i = 0
    while not solved:

        # 3. Evolve for <x> generations:
        for generation in range(config.generations * i, config.generations * (i + 1)):
            config.generation = generation

            # 3.1 Select some patterns for mutation. Tournament
            selected = selection.tournament(patterns, size=int(len(patterns) / 2))

            # 3.2 Perform Mutations + Crossover on selected patterns
            mutations, crossovers = selection.divide(selected)
            patterns = patterns + \
                       mutator.apply(mutations) + \
                       crossover.apply(crossovers)

            # 3.3 Evaluate new patterns. Fitness calculation
            nets = recombine.combine(patterns, compute_capacity, config.min_size, config.max_size, include_optimal=True)
            nets = workers.start(nets, config)
            patterns = evaluation.inherit_results(patterns, nets)

            # 3.4 Rank all patterns using MOO. Diversity in position, 2D vs 1D, scores ++
            patterns = nsga_ii(patterns, moo_objectives, domination_operator, config)

            # 3.5 Evolution of the fittest. Elitism
            patterns = patterns[:config.population_size]

            # 3.6 Feedback:
            print(f"--> Generation {generation} Leaderboards")
            generation_finished(patterns, config, "    - Patterns:")
            generation_finished(patterns, config, "    - Neural networks:")
            config.results.store_generation(patterns, generation)
            config.results.store_generation(nets, generation)

        # To finish up, the best combination of patterns needs to be returned and trained for
        # much longer than what they are during fitness evaluation. The previous steps should only
        # be used for verifying that the combination of layers is good.
        #
        # This might need to be tried multiple times. When a good result is gotten, the algorithm should
        # stop and return the final structure with trained weights.

        print("Testing best combination of patterns")

        # Changing settings of training steps:
        original_training_settings = copy.deepcopy(config.training)
        config.training.use_restart = False
        config.training.fixed_epochs = True
        config.training.epochs = 300

        # Finding the best combined network:@
        config.type = "ea-nas"
        nets.sort(key=weighted_overfit_score(config), reverse=True)
        config.type = "PatternNets"
        best_net = nets[0]

        # Performing training step:
        best_net = workers.start([best_net], config)[0]

        # Reset settings and return:
        config.training = original_training_settings

        if best_net.validation_fitness[-1] >= config.training.acceptable_scores:
            print("Found good network! ")
            solved = True

        config.type = "ea-nas"
        generation_finished([best_net], config, "--> Found final solution:")
        config.type = "PatternNets"
        patterns = evaluation.inherit_results(patterns, nets)
        i += 1
