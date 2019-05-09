import pickle
from mpi4py.futures import MPIPoolExecutor
from src.training import prepare_training as trainer


def pack_args(population, config):
    server_job_args = []
    config_str = pickle.dumps(config)
    # jobs_per_server = int(len(population) / len(config.servers))
    epochs = config.training.epochs if config.training.fixed_epochs else 20
    for i, individ in enumerate(population):
        server_id = i % len(config.servers)
        server_job_args += [(
            pickle.dumps(population[i]),
            config_str,
            epochs,
            server_id,
            0,
            i
        )]
    return server_job_args


def launch_with_MPI_futures(population, config):
    for individ in population:
        individ.failed = False

    args = pack_args(population, config)

    # Calculating TensorFlow job size:
    print(
        "--> Fitness calculation using MPI Pool executor.\n" +
        f"    - Jobs running: {len(args)} - on {len(config.servers)} servers"
    )

    # Running TensorFlow jobs with MPI
    with MPIPoolExecutor() as executor:
        results = [result for result in executor.map(trainer.run, args)]
        executor.shutdown(wait=True)

    results = [x for x in results if not x.failed]
    for individ in results:
        del individ.failed

    # Exceptions may occur inside the async training loop.
    # The failed solutions will be discarded:
    print(f"    - Entire population trained. {len(population) - len(results)}/{len(population)} failed.")
    return results
