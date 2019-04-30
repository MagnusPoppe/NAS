from mpi4py import futures
from mpi4py.futures import MPIPoolExecutor

from src.training import prepare_training as trainer
from src.jobs.common_mp import pack_args


def launch_with_MPI_futures(population, config):
    args = [
        pack_args(population, id, config)
        for id, server in enumerate(config.servers)
    ]
    with MPIPoolExecutor() as executor:
        return executor.map(trainer.run, args)
