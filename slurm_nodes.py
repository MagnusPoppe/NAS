import sys
import time
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
        while line != '':
            line = out.readline()
            output += line

        if output != "":
            print("{} reads:\n{}".format(name, output))
        return output

    ssh_process = ps.Popen(
        args=['ssh', "-T", '{}'.format(server)],
        stdin=ps.PIPE,
        stdout=ps.PIPE,
        stderr=ps.PIPE,
        universal_newlines=True,
        bufsize=0
    )
    for command in commands:
        ssh_process.stdin.write(command + "\n")
    ssh_process.stdin.write('logout\n')

    while ssh_process.poll() is not None:
        time.sleep(0.01)

    stdout = read(ssh_process.stdout, name="Stdout")
    stderr = read(ssh_process.stderr, name="Stderr")
    ssh_process.stdin.close()
    ssh_process.stdout.close()
    ssh_process.stderr.close()
    return stdout, stderr

# config_json = "./datasets/cifar10-slurm.py"  # sys.argv[2]
arg = sys.argv[1]
servers = slurm_nodelist_to_list(arg)
for address in servers:
    print("Executing nvidia-smi on node", address)
    exec_remote(address, commands=[
        'python -c ' +
        "'import subprocess; print(subprocess.check_output([\"nvidia-smi\", \"-L\"]))'"
    ])
