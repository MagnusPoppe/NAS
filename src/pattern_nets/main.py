import sys
import random

from src.configuration import Configuration
from src.output import generation_finished
from src.pattern_nets.initialization import initialize_patterns
from src.pattern_nets import recombine, evaluation, selection, mutator, crossover
if len(sys.argv) == 3 and sys.argv[2] == "mock":
    import src.training.mock as workers
else:
    import src.jobs.job_initializer as workers


def main(config: Configuration):
    # How many nets can be trained for each generation?
    compute_capacity = sum([dev.concurrency for server in config.servers for dev in server.devices])
    compute_capacity *= 2

    # 1. Initialize population:
    patterns = initialize_patterns(config.population_size)

    # 2. Evaluation of initial population. Fitness calculation
    nets = recombine.combine(patterns, compute_capacity, config.min_size, config.max_size)
    nets = workers.start(nets, config)
    patterns = evaluation.apply_results(patterns, nets)
    generation_finished(nets, f"--> Initialization Complete:")

    # 3. Evolve for x generations:
    for generation in range(config.generations):
        # 3.1 Select some patterns for mutation. Tournament
        selected = selection.tournament(patterns, size=int(len(patterns)/2))

        # 3.2 Perform Mutations + Crossover on selected patterns
        mutations, crossovers = selection.divide(selected)
        patterns = patterns + mutator.apply(mutations) + crossover.apply(crossovers)

        # 3.3 Evaluate new patterns. Fitness calculation
        nets = recombine.combine(patterns, compute_capacity, config.min_size, config.max_size)
        nets = workers.start(nets, config)
        patterns = evaluation.apply_results(patterns, nets)

        # 3.4 Rank all patterns. MOOEA. Diversity in position, 2D vs 1D, scores ++
        pass

        # 3.5 Evolution of the fittest. Elitism
        random.shuffle(patterns)
        patterns = patterns[:config.population_size]

        # 3.6 Feedback:
        generation_finished(nets, f"--> Generation {generation} Leaderboards:")
