import sys

from src.MOOA.NSGA_II import nsga_ii
from src.configuration import Configuration
from src.output import generation_finished, print_population
from src.pattern_nets.initialization import initialize_patterns
from src.pattern_nets import recombine, evaluation, selection, mutator, crossover
from src.pattern_nets import moo_objectives as moo

if len(sys.argv) == 3 and sys.argv[2] == "mock":
    import src.training.mock as workers
else:
    import src.jobs.job_initializer as workers


def main(config: Configuration):
    # 0.1 How many nets can be trained for each generation?
    compute_capacity = sum([dev.concurrency for server in config.servers for dev in server.devices])

    # 0.2 Initializing multi objective optimisation sorting:
    moo_objectives = moo.classification_objectives(config)
    domination_operator = moo.classification_domination_operator(
        moo.classification_objectives(config)
    )

    # 1. Initialize population:
    patterns = initialize_patterns(config.population_size)
    print_population(patterns)

    # 2. Evaluation of initial population. Fitness calculation
    nets = recombine.combine(patterns, compute_capacity, config.min_size, config.max_size)
    nets = workers.start(nets, config)
    patterns = evaluation.inherit_results(patterns, nets)

    generation_finished(patterns, config, f"--> Initialization Complete:")

    # 3. Evolve for x generations:
    for generation in range(config.generations):

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
        generation_finished(patterns, config, f"--> Generation {generation} Leaderboards:")
