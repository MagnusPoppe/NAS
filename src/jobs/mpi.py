import pickle
from mpi4py import MPI
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

    mpi_size = MPI.Comm.Get_size(MPI.COMM_WORLD) -1
    results = []
    for i in range(0, len(args), mpi_size):
        # Calculating TensorFlow job size:
        jobs = i + mpi_size if i + mpi_size < len(args) else len(args)
        print(
            "--> Starting MPI Pool executor. " +
            f"Jobs running {i}-{jobs} of {len(args)} on {len(config.servers)} servers"
        )

        # Running TensorFlow jobs with MPI
        with MPIPoolExecutor(max_workers=mpi_size) as executor:
            results += [result for result in executor.map(trainer.run, args[i:jobs])]
            executor.shutdown(wait=True)

    # Exceptions may occur inside the async training loop.
    # The failed solutions will be discarded:
    original = len(results)
    results = [individ for individ in results if not individ.failed]
    filtered = len(results)
    print(f"--> Entire population trained. {original-filtered}/{original} failed.")
    for individ in results:
        del individ.failed

    return results
