from src.buildingblocks.module import Module
from src.configuration import Configuration
from src.jobs.MP import run_jobs
from src.jobs.execnet import launch_with_execnet
from src.jobs.mpi import launch_with_MPI_futures


def start(population: [Module], config: Configuration) -> [Module]:
    """
    Starts the training rutines for all of the neural networks.

    """
    # CASE 1, Runs locally:
    if config.MPI:
        return launch_with_MPI_futures(population, config)
    elif len(config.servers) > 1 and config.servers[0].type == "remote":
        return launch_with_execnet(population, config)
    else:
        return run_jobs(population, server_id=0, config=config)
