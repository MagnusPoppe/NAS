import time
import subprocess as ps


def read(out):
    line = out.readline()
    output = line
    while line != '':
        line = out.readline()
        output += line
    return output


def exec_remote(server: dict, commands: [str]) -> (str, str):
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

    stdout = read(ssh_process.stdout)
    stderr = read(ssh_process.stderr)
    ssh_process.stdin.close()
    ssh_process.stdout.close()
    ssh_process.stderr.close()
    return stdout, stderr


def rsync(source: str, dest: str, server: dict, to_source: bool = True):
    exec_remote(server, ['mkdir -r {}'.format(dest)])
    server_address = server['address']
    if to_source:
        source = "/".join(source.split("/")[:-1])
        command = ['rsync', '-r', '-azh',  f"{server_address}:{dest}", source]
    else:
        dest = "/".join(dest.split("/")[:-1])
        command = ['rsync', '-r', '-azh', source, f"{server_address}:{dest}"]

    process = ps.Popen(
        args=command,
        stdin=ps.PIPE,
        stdout=ps.PIPE,
        stderr=ps.PIPE,
        universal_newlines=True,
        bufsize=0
    )
    while process.poll() == None:
        time.sleep(0.01)

    err = read(process.stderr)
    if err != "": raise Exception(err)

    process.stdin.close()
    process.stdout.close()
    process.stderr.close()

def _par_rsync(args):
    rsync(*args)

def rsync_parallel(transfer_args: [(str, str, dict, bool)]):
    """
    Runs the rsync function in parallel
    :param transfer_args: formatted as arguments to rsync function above
    """
    import multiprocessing as mp
    if transfer_args:
        pool = mp.Pool()
        pool_workers = pool.map_async(_par_rsync, transfer_args)
        pool.close()
        _ = pool_workers.get()  # Waits for finish
