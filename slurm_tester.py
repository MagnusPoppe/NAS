import os
from mpi4py import MPI

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import time
    time.sleep(rank/2)
    print(f"[{rank}/{size}] is {'MAIN  ' if rank == 0 else 'WORKER'}", end=" ", flush=True)
    os.system("hostname")

