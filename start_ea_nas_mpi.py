import pickle
import sys

from mpi4py import MPI

from src.configuration import Configuration
import src.ea_nas.main as ea_nas

try:
    import setproctitle

    setproctitle.setproctitle("EA-NAS-EVOLVE")
except ImportError:
    pass

MAIN_RANK = 0


def distribute_with_mpi(workloads):
    # Preparation
    comm = MPI.COMM_WORLD

    # Distributing jobs:
    for recv_rank, workload in enumerate(workloads):
        print(f"\n\t[{comm.Get_rank()}] Sending work to {recv_rank+1} tagged {0}")
        comm.send(workload, dest=recv_rank+1, tag=0)

    # Receiving results:
    new_population = []
    for sender_rank in range(1, comm.Get_size()):
        print(f"\t[{comm.Get_rank()}] Waiting for {sender_rank} to send back work")
        new_population += comm.recv(source=sender_rank, tag=0)

    return new_population


def net_trainer_main(comm: MPI.Comm, config: Configuration):
    from src.jobs.TF_launcher import run_jobs
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
        population = run_jobs(population, rank-1, config=config, verbose=False)

        # Return results
        print(f"\n\t[{comm.Get_rank()}] Sending work to {MAIN_RANK} tagged {0}")
        comm.send(population, dest=MAIN_RANK, tag=0)


def main():
    # Setting up MPI:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Setting up config with MPI:
    config = Configuration.from_json(sys.argv[1])
    config.MPI = True

    assert len(config.servers) == comm.Get_size() - 1, "Should be as many processes as servers..."

    # Evolving or training based on rank:
    if rank == MAIN_RANK:
        ea_nas.run(config)
        for other_rank in range(1, comm.Get_size()):
            comm.send("stop", dest=other_rank, tag=0)
    else:
        net_trainer_main(comm, config)
    MPI.Finalize()


if __name__ == '__main__':
    main()
