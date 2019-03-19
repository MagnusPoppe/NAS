from src.configuration import Configuration
from src.pattern.initialization import initialize_patterns
from src.pattern import recombination
import src.jobs.job_initializer as workers

def main(selection: callable, config: Configuration):
    # 1. Initialize Genepool (Population):
    patterns = initialize_patterns(config.population_size)

    # 2. Evaluation of initial Genepool. Fitness calculation
    modules = recombination.combine(patterns, count=config.population_size)
    modules = workers.start([pattern.flatten() for pattern in modules], config)

    # 3. Evolve for x generations:
    for generation in range(config.generations):
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

