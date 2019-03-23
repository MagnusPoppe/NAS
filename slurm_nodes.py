import os
import sys
import time
import json
import subprocess as ps


def slurm_nodelist_to_list(_input):
    """
    Takes the slurm compute node id string and splits it into
    separate node adresses.
    :param _input: $SLURM_JOB_NODELIST env variable
    :return: list of server addresses
    """
    if ",c" in _input:
        new_input = _input.split(",c")
        _input = [new_input[0]] + ["c" + x for x in new_input[1:]]
    else:
        _input = [_input]

    server_addresses = []
    for inn in _input:

        if not "[" in inn:
            server_addresses += [inn]
            continue

        root, raw = inn.split("[")
        raw_ids = raw.strip("]").split(",")
        ids = []

        for id in raw_ids:
            if "-" in id:
                x = id.split("-")
                assert len(x) == 2  # Should fail here if string has bad format
                ids += [y for y in range(int(x[0]), int(x[1]) + 1)]
            else:
                ids += [int(id)]

        server_addresses += [root + str(id) for id in ids]
    server_addresses.sort()
    return server_addresses


def exec_remote(server, commands):
    def read(out, name="STDOUT"):
        line = out.readline()
        output = line
        while line != "":
            line = out.readline()
            output += line
        return output

    ssh_process = ps.Popen(
        args=["ssh", "-T", "{}".format(server)],
        stdin=ps.PIPE,
        stdout=ps.PIPE,
        stderr=ps.PIPE,
        universal_newlines=True,
        bufsize=0,
    )
    for command in commands:
        ssh_process.stdin.write(command + "\n")
    ssh_process.stdin.write("logout\n")

    while ssh_process.poll() is not None:
        time.sleep(0.01)

    stdout = read(ssh_process.stdout, name="Stdout")
    stderr = read(ssh_process.stderr, name="Stderr")
    ssh_process.stdin.close()
    ssh_process.stdout.close()
    ssh_process.stderr.close()
    return stdout, stderr


arg = sys.argv[1]
config_json = sys.argv[2]
with open(config_json, "r") as f:
    config = json.load(f)

servers = []
for address in slurm_nodelist_to_list(arg):
    out, err = exec_remote(
        address,
        commands=[
            'python -c \'import subprocess; print(subprocess.check_output(["nvidia-smi", "-L"]))\''
        ],
    )
    # if err:
    #     raise Exception(err)
    server = {
        "name": "EPIC-" + address,
        "type": "remote",
        "cwd": os.getcwd(),
        "address": address,
        "python": "~/ea-nas/venv/bin/python",
        "devices": [],
    }
    gpu_strings = [g for g in out.split("\n") if g]
    for gpu_line in gpu_strings:
        gpu_id = gpu_line.split(":")[0].split(" ")[1]
        concurreny = 2 if "Tesla V100" in gpu_line else 1
        server["devices"] += [
            {
                "device_str": f"/GPU:{gpu_id}",
                "allow gpu memory growth": True,
                "memory per process": 1 / concurreny,
                "concurrency": concurreny,
            }
        ]
    servers += [server]

config["servers"] = servers
with open(config_json, "w") as f:
    json.dump(config, f)
