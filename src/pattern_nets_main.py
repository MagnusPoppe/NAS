from src.configuration import Configuration
from src.pattern_nets.initialization import initialize_patterns
from src.pattern_nets import recombination
import src.jobs.job_initializer as workers

def main(selection: callable, config: Configuration):
    # 1. Initialize Genepool (Population):
    patterns = initialize_patterns(config.population_size)

    # 2. Evaluation of initial Genepool. Fitness calculation
    nets = recombination.combine_random(patterns, num_nets=3)
    nets = workers.start([pattern.flatten() for pattern in nets], config)

    # 3. Evolve for x generations:
    for generation in range(config.generations):
        nets = recombination.combine_random(patterns, num_nets=3)
        nets = workers.start([pattern.flatten() for pattern in nets], config)

        # 3.1 Select some patterns for mutation. Tournament
        pass

        # 3.2 Perform Mutations + Crossover on selected patterns
        pass

        # 3.3 Evaluate new patterns. Fitness calculation
        pass

        # 3.4 Rank all patterns. MOOEA. Diversity in position, 2D vs 1D, scores ++
        pass

        # 3.5 Evolution of the fittest. Elitism
        pass

