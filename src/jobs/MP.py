import multiprocessing as mp

from src.jobs.common_mp import pack_args
from src.training import prepare_training as trainer


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
