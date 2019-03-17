import os
import pickle
import time
import execnet

from src.buildingblocks.module import Module
from src.configuration import Configuration
from src.jobs.TF_launcher import run_jobs
from src.jobs.ssh import rsync_parallel
import src.jobs.launch_remote as run_jobs_remote


def time_str(started, ts, offset):
    timediff = time.time() - started
    return f".{' '*(len(ts) + 4 - offset)}Finished in {timediff:.2f} sec"

def create_target(model_path, server_id, config):
    exploded = model_path.split("/")
    split_point = -1
    for i, part in enumerate(exploded):
        if part == "results":
            split_point = i
            break
    else:
        raise NotADirectoryError("Could not find \"results\" in model path")

    return os.path.join(config.servers[server_id].cwd, "/".join(exploded[split_point:]))


def transfer_args(workloads, config, to_source=False) -> [(str, str, dict, bool)]:
    args = []
    for server_id, sub_population in enumerate(workloads):
        server = config.servers[server_id]
        for individ in sub_population:
            m_path = individ.absolute_save_path(config)
            args += [(m_path, create_target(m_path, server_id, config), pickle.dumps(server), to_source)]
            if individ.predecessor:
                m_path = individ.predecessor.absolute_save_path(config)
                args += [(m_path, create_target(m_path, server_id, config), pickle.dumps(server), to_source)]

    return args


def start(population: [Module], config: Configuration) -> [Module]:
    """
    Starts the training rutines for all of the neural networks.

    case 1: Jobs are running locally
        1. Run jobs...

    case 2: Jobs are running on a distributed cluster using SSH
        1. Divide up tasks for each node in cluster
        2. Transfer necessary data to each node (model.h5 files)
        3. Start the algorithm
        4. Gather direct results
        5. Gather data generated on each server
        6. return results
    """
    # CASE 1, Runs locally:
    if len(config.servers) == 1 and config.servers[0].type == "local":
        start = time.time()
        new_population = run_jobs(population, server_id=0, config=config)
        print(f"    - Time used {(time.time() - start):.2f} sec")
        return new_population

    # CASE 2, Runs distributed:
    # 1. Divide up tasks for each node in cluster
    magic_number = int(len(population) / len(config.servers))
    workloads = [(population[i:i + magic_number]) for i in range(len(config.servers))]

    # For cosmetics in output window:
    avg_work = sum(len(w) for w in workloads) / len(workloads)
    training_str = f"--> Training on {len(workloads)} servers. {avg_work} networks/server"

    # 2. Transfer necessary data to each node (model.h5 files)
    start = time.time()
    print("--> Distributing files to hosts", end="")
    rsync_parallel(transfer_args(workloads, config, to_source=False))
    print(time_str(start, training_str, offset=31))

    # 3. Start the algorithm
    start = time.time()
    print(training_str, end="", flush=True)
    comms = []
    for i, server in enumerate(config.servers):
        gw = execnet.makegateway(f"ssh={server.address}//python={server.python}//chdir={server.cwd}")
        ch = gw.remote_exec(run_jobs_remote)
        ch.send(pickle.dumps((workloads[i], i, config)))
        comms += [(gw, ch)]

    # 4. Gather direct results
    new_population = []
    for gw, ch in comms:
        received = ch.receive()
        result = pickle.loads(received)
        if isinstance(result, Exception):
            raise result
        new_population += result

    mch = execnet.MultiChannel([ch for _, ch in comms])
    mch.waitclose()
    for gw, _ in comms: gw.exit()
    print(time_str(start, training_str, offset=0))

    # 5. Gather files produced on remote server
    start = time.time()
    print("--> Gathering files from hosts", end="")
    rsync_parallel(transfer_args(workloads, config, to_source=True))
    print(time_str(start, training_str, offset=30))

    # 6. return results
    return new_population
