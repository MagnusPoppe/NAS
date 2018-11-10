import pickle
import os
import execnet
import time
from datasets import cifar10

# TODO: Check how many GPUs are available in total and manage them separatly for parallel training
# TODO:

from firebase.upload import update_status, upload_population

packages = []

def rsync(source, dest, server, to_source=True):
    from src.jobs import ssh
    ssh.exec_remote(server, ['mkdir -r {}'.format(dest)])

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

def receive(payload):
    global packages
    packages += [payload]
    print("=", end="", flush=True)
    # TODO: Start new process here!
    # Idea: spawn all processes and let them idle on
    # channel.recieve() serverside. When a process is
    # complete, send new inputs for next channel. A
    # new job will start ...


def launch_trainers(population, config):
    global packages
    packages = []
    server_config = config['servers'][0]

    print("--> Training on servers {} |".format(server_config['name']), end="", flush=True)
    started = time.time()
    trained = 0
    concurrency = 1
    for i in range(0, len(population), concurrency):
        channels = []
        gateways = []

        # Starting training sessions:
        for individ in population[i:i+concurrency]:

            # Connecting to server:
            gateway = get_local_gateway(server_config)

            # Syncronizing models:
            upload_models(individ, config, server_config)

            # Executing script and getting process id:
            channel = gateway.remote_exec(cifar10)
            channel.setcallback(receive)

            # Sending data to be processed and recieving fitness:
            channel.send((pickle.dumps(individ), pickle.dumps(config), pickle.dumps(server_config)))

            with open(individ.get_relative_module_save_path(config) + "/genotype.obj", "wb") as f:
                pickle.dump(individ, f)

            gateways += [gateway]
            channels += [channel]

        update_status("Training {} for {} epochs per op ( {}/{} models complete )".format(
            [individ.ID for individ in population[i:i+concurrency]],
            config['epochs'],
            trained,
            len(population)
        ))

        mch = execnet.MultiChannel(channels)
        mch.waitclose()
        for gw in gateways:
            gw.exit()

    # Ending training sessions:
    for i, individ in enumerate(population):
        # Setting results:
        individ.fitness, individ.saved_model, individ.model_image_link = packages[i]

        # Syncing model files
        download_models(individ, config, server_config)

    upload_population(population)
    packages = []
    print("| (Elapsed time: {} sec)".format(time.time()-started))