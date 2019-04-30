import pickle
import sys

from mpi4py import MPI

from src.configuration import Configuration
import src.ea_nas.main as ea_nas
from src.jobs.MP import run_jobs

try:
    import setproctitle

    setproctitle.setproctitle("NAS-EVOLVE")
except ImportError:
    pass

MAIN_RANK = 0


def distribute_with_mpi(workloads, config):
    # Preparation
    comm = MPI.COMM_WORLD

    # Distributing jobs:
    for recv_rank, workload in enumerate(workloads):
        if recv_rank == MAIN_RANK: continue
        print(f"\n\t[{comm.Get_rank()}] Sending work to {recv_rank} tagged {0}")
        comm.send(workload, dest=recv_rank, tag=0)

    new_population = run_jobs(workloads[0], 0, config=config, verbose=False)

    # Receiving results:
    for sender_rank in range(1, comm.Get_size()):
        print(f"\t[{comm.Get_rank()}] Waiting for {sender_rank} to send back work")
        new_population += comm.recv(source=sender_rank, tag=0)

    return new_population


def net_trainer_main(comm: MPI.Comm, config: Configuration):
    rank = comm.Get_rank()
    while True:
        # Receive nets for training
        print(f"\t[{rank}] Waiting for work...")
        recieved = comm.recv(source=MAIN_RANK, tag=0)

        # Checking for kill:
        if isinstance(recieved, str) and recieved == "stop": break
        else: population = recieved

        print(f"\t[{rank}] Work received!")

        # Run training
        population = run_jobs(population, rank, config=config, verbose=False)

        # Return results
        print(f"\n\t[{comm.Get_rank()}] Sending work to {MAIN_RANK} tagged {0}")
        comm.send(population, dest=MAIN_RANK, tag=0)


def main():
    # Setting up MPI:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Setting up config with MPI:
    config = Configuration.from_json(sys.argv[1])
    config.MPI = True
    config.type = "ea-nas"

    import subprocess as ps
    host = ps.check_output("hostname")
    print(f"[{rank}/{size}] Reporting in from {host}")
    if rank == MAIN_RANK:
        from src.output import print_config_stats
        print_config_stats(config)
        print(f"Checking if {len(config.servers)} == {size}")
    assert len(config.servers) == size, "Should be as many processes as servers..."

    # Evolving or training based on rank:
    if rank == MAIN_RANK:
        try:
            ea_nas.run(config)
        except Exception as e:
            for other_rank in range(1, size):
                comm.send("stop", dest=other_rank, tag=0)
            raise e
        for other_rank in range(1, size):
            comm.send("stop", dest=other_rank, tag=0)
    else:
        net_trainer_main(comm, config)
    MPI.Finalize()


if __name__ == '__main__':
    main()
