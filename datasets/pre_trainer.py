import pickle
import os
import execnet
import time
from datasets import cifar10

# TODO: Check how many GPUs are available in total and manage them separatly for parallel training
# TODO:
from firebase.upload import update_status
from modules.module import Module


def execute_remote(server, commands):
    import subprocess as ps
    ssh_process = ps.Popen(
        args=['ssh', '{}@{}'.format(server['username'], server['address'])],
        stdin=ps.PIPE,
        stdout=ps.PIPE,
        stderr=ps.PIPE,
        universal_newlines=True,
        bufsize=0
    )
    for command in commands:
        ssh_process.stdin.write(command + "\n")
    ssh_process.stdin.write('logout\n')
    # print(ssh_process.stdout)
    # print(ssh_process.stderr)
    ssh_process.stdin.close()

def rsync(source, dest, server, to_source=True):
    execute_remote(server, ['mkdir -r {}'.format(dest)])

    import subprocess as ps
    if to_source:
        source = "/".join(source.split("/")[:-1])
        command = ['rsync', '-r', '-azh', server['username']+'@'+server['address']+':'+dest, source]
    else:
        dest = "/".join(dest.split("/")[:-1])
        command = ['rsync', '-r', '-azh', source, server['username'] + '@' + server['address'] + ':' + dest]

    ps.Popen(
        args=command,
        stdin=ps.PIPE,
        stdout=ps.PIPE,
        stderr=ps.PIPE,
        universal_newlines=True,
        bufsize=0
    )


def get_local_gateway(server):
    return execnet.makegateway("ssh={ssh}//python={py}//chdir={dir}".format(
        ssh="{}@{}".format(server['username'], server['address']),
        py="python",
        dir=server['cwd'],
    ))

def upload_models(individ, config, server_config):
    local_path = os.path.join(os.getcwd(), individ.get_relative_module_save_path(config))
    server_path = os.path.join(server_config['cwd'], individ.get_relative_module_save_path(config))
    rsync(local_path, server_path, server_config, to_source=False)
    if individ.predecessor:
        local_path = os.path.join(os.getcwd(), individ.predecessor.get_relative_module_save_path(config))
        server_path = os.path.join(server_config['cwd'], individ.predecessor.get_relative_module_save_path(config))
        rsync(local_path, server_path, server_config, to_source=False)

def download_models(individ, config, server_config):
    rsync(
        os.path.join(os.getcwd(), individ.get_relative_module_save_path(config)),
        os.path.join(server_config['cwd'], individ.get_relative_module_save_path(config)),
        server_config,
        to_source=True
    )


def launch_remote_training(individ, config, server_config):

    # Connecting to server:
    gateway = get_local_gateway(server_config)

    # Syncronizing models:
    upload_models(individ, config, server_config)

    # Executing script and getting process id:
    channel = gateway.remote_exec(cifar10)
    pid = channel.receive()

    # Sending data to be processed and recieving fitness:
    channel.send((pickle.dumps(individ), pickle.dumps(config)))
    with open(individ.get_relative_module_save_path(config) + "/genotype.obj", "wb") as f:
        pickle.dump(individ, f)
    return channel, pid, individ

def collect_remote_results(channel, pid, individ, config, server_config):
    # Waiting for program to finish and for results:
    individ.fitness, model_path_remote = channel.receive()
    individ.saved_model = model_path_remote

    # Shutting down process after finishing:
    channel.waitclose()

    # KILL to free memory:
    execute_remote(
        server=server_config,
        commands=['kill {}'.format(pid)]
    )

    # Syncing model files
    download_models(individ, config, server_config)

def launch_trainers(population, config):
    server_config = {
        'username': 'magnus',
        'address': '192.168.1.10',
        'cwd': '/home/magnus/remote/EA-architecture-search'
    }

    print("--> Training on servers {} |".format(server_config['address']), end="", flush=True)
    started = time.time()
    trained = 0
    concurrency = 1
    while trained < len(population):
        channels = []


        # Starting training sessions:
        for i in range(concurrency):
            if trained + i > len(population): break
            individ = population[trained + i]  # type: Module
            channels += [launch_remote_training(individ, config, server_config)]

        update_status("Training {} for {} epochs per op ( {}/{} models complete )"
                      .format([individ.ID for _, _, individ in channels], config['epochs'], trained, len(population)))


        # Ending training sessions:
        for i in range(concurrency):
            if trained + i > len(population):
                trained += 1
                break
            collect_remote_results(*channels[i], config=config, server_config=server_config)
            trained += 1
            print("=", end="", flush=True)
        time.sleep(1)
    print("| (Elapsed time: {} sec)".format(time.time()-started))