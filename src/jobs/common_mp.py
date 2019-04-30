import pickle
import sys
from src.configuration import Configuration


def pack_args(population, server_id, config: Configuration):
    """ Compiles a list of arguments for parallel training """
    config_str = pickle.dumps(config)

    # Each server gets a portion of the jobs:
    server_job_args = [[] for _ in range(len(config.servers[server_id].devices))]

    # Calculating size of each job:
    sized = []
    for ind in population:
        epochs = config.training.epochs \
            if config.training.fixed_epochs \
            else int(ind.number_of_operations() * config.training.epochs)
        sized += [epochs if epochs > 0 else 1]
    total_epochs = sum(sized)

    # Create balanced workloads for each process by estimate:
    for i in range(len(population)):
        dev_id = i % len(config.servers[server_id].devices)
        job_id = sized.index(max(sized))
        epochs = int(sized[job_id])

        # Compiling list of arguments:
        server_job_args[dev_id] += [(
            pickle.dumps(population[job_id]),
            config_str,
            epochs,
            server_id,
            dev_id,
            job_id
        )]
        sized[job_id] = - sys.maxsize

    return server_job_args, total_epochs
