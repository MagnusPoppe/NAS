import time
import pickle
import execnet
import os

from src.jobs.file_sync import rsync_parallel


def time_str(started, ts, offset):
    timediff = time.time() - started
    return f".{' ' * (len(ts) + 4 - offset)}Finished in {timediff:.2f} sec"


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


def launch_with_execnet(population, config):
    """
    Jobs are running on a distributed cluster using SSH
        1. Divide up tasks for each node in cluster
        2. Transfer necessary data to each node (model.h5 files)
        3. Start the algorithm
        4. Gather direct results
        5. Gather data generated on each server
        6. return results
    """
    # 1. Divide up tasks for each node in cluster
    magic_number = int(len(population) / len(config.servers))
    workloads = [(population[i:i + magic_number]) for i in range(len(config.servers))]

    # 2. Transfer necessary data to each node (model.h5 files)
    if not config.MPI and config.servers[0].type == "remote":
        rsync_parallel(transfer_args(workloads, config, to_source=False))

    # 3. Start the algorithm
    new_population = distribute_with_execnet(config, workloads)

    # 5. Gather files produced on remote server
    rsync_parallel(transfer_args(workloads, config, to_source=True))

    # 6. return results
    return new_population


def distribute_with_execnet(config, workloads):
    import src.jobs.execnet_recieve as run_jobs_remote

    # 3. Run jobs on remote servers:
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

    # 5. Waiting for results and shutting down:
    mch = execnet.MultiChannel([ch for _, ch in comms])
    mch.waitclose()
    for gw, _ in comms: gw.exit()

    return new_population
