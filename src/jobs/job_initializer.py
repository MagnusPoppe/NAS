import os
import pickle
import time

import execnet

from src.buildingblocks.module import Module
from src.configuration import Configuration
from src.jobs.TF_launcher import run_jobs
from src.jobs.ssh import rsync_parallel
import src.jobs.launch_remote as run_jobs_remote


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


def generate_transfer_args(workloads, config, to_source=False) -> [(str, str, dict, bool)]:
    def locate_model(individ: Module) -> str or None:
        """ Checks if a model is created and returns path if exists """
        path = individ.relative_save_path(config)
        model_path = os.path.join(path, "model.h5")
        return path if os.path.isfile(model_path) else None

    args = []
    for server_id, sub_population in enumerate(workloads):
        server = config.servers[server_id]
        for individ in sub_population:
            m_path = locate_model(individ)
            if m_path:
                args += [(m_path, create_target(m_path, server_id, config), server, to_source)]
            if individ.predecessor:
                m_path = locate_model(individ.predecessor)
                if m_path:
                    args += [(m_path, create_target(m_path, server_id, config), server, to_source)]
    return args


def transfer_args(workloads, config, to_source=False) -> [(str, str, dict, bool)]:
    args = []
    for server_id, sub_population in enumerate(workloads):
        server = config.servers[server_id]
        for individ in sub_population:
            m_path = individ.absolute_save_path(config)
            args += [(m_path, create_target(m_path, server_id, config), pickle.dumps(server), to_source)]
    return args

def start(population: [Module], config: Configuration) -> [Module]:
    """
    Starts the training rutines for all of the neural networks.
    case 1: Jobs are running locally
    case 2: Jobs are running on a distributed cluster using SSH

    Procedure for case 2:
    1. Divide up tasks for each node in cluster
    2. Transfer necessary data to each node (model.h5 files)
    3. Start the algorithm
    4. Gather direct results
    5. Gather model.h5 files
    6. return results
    """
    # CASE 1, Runs locally:
    if len(config.servers) == 1 and config.servers[0].type == "local":
        return run_jobs(population, server_id=0, config=config)

    # CASE 2, Runs distributed:
    # 1. Divide up tasks for each node in cluster
    magic_number = int(len(population) / len(config.servers))
    workloads = [(population[i:i + magic_number]) for i in range(len(config.servers))]

    # 2. Transfer necessary data to each node (model.h5 files)
    start = time.time()
    print("--> Distributing files to hosts", end="")
    rsync_parallel(transfer_args(workloads, config, to_source=False))
    print(f". Time: {time.time() - start} sec")

    # 3. Start the algorithm
    start = time.time()
    print("--> Training networks remotly", end="", flush=True)
    comms = []
    for i, server in enumerate(config.servers):
        gw = execnet.makegateway(f"ssh={server.address}//python={server.python}//chdir={server.cwd}")
        ch = gw.remote_exec(run_jobs_remote)
        ch.send(pickle.dumps((workloads[i], i, config)))
        comms += [(gw, ch)]

    # 4. Gather direct results
    new_population = []
    for gw, ch in comms:
        sub_population_str = ch.receive()
        result = pickle.loads(sub_population_str)
        if isinstance(result, Exception):
            raise result
        new_population += result

    mch = execnet.MultiChannel([ch for _, ch in comms])
    mch.waitclose()
    for gw, _ in comms: gw.exit()
    print(f". Time: {time.time() - start} sec")

    # 5. Gather model.h5 files
    start = time.time()
    print("--> Gathering files from hosts", end="")
    rsync_parallel(transfer_args(workloads, config, to_source=True))
    print(f". Time: {time.time() - start} sec")

    # 6. return results
    return new_population



if __name__ == '__main__':
    import LAB.common as fn
    strains = fn.load_all_modules_from_run("./results/8x2")
    population = []
    for individs in strains.values():
        population += individs
    config = Configuration.from_json("./datasets/cifar-10-mp.json")
    start(population, config)
