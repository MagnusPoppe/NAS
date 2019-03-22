import pickle
import execnet
import os
from src.jobs.TF_launcher import run_jobs

def distribute_with_execnet(config, workloads):
    import src.jobs.launch_remote as run_jobs_remote

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

if __name__ == '__channelexec__':
    args = channel.receive()
    population, server_id, config = pickle.loads(args)
    try:
        new_population = run_jobs(population, server_id, config=config)
        if not any(len(x.report) > 0 or len(x.fitness) > 0 for x in new_population):
            raise Exception("New population has not trained...")
        channel.send(pickle.dumps(new_population))
    except Exception as e:
        channel.send(pickle.dumps(e))
