import pickle
import sys
import time
import multiprocessing as mp

from src.configuration import Configuration
from src.training import prepare_training as trainer


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


def run_jobs(population, server_id, config, verbose=False):
    server_args, epochs = pack_args(population, server_id, config)
    print(f"--> Running {epochs} epoch(s) of training for {len(population)} phenotype(s)")

    # Spawning jobs:
    pools = []
    pool_res = []
    for id, device in enumerate(config.servers[server_id].devices):
        pool = mp.Pool(processes=device.concurrency)
        pool_res += [
            pool.map_async(trainer.run, server_args[id])
        ]
        pools += [pool]

    # Awaiting results:
    results = []
    for pool, res in zip(pools, pool_res):
        pool.close()
        results += res.get()

    return results
