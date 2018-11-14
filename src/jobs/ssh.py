import time


def exec_remote(server, commands):
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

    ssh_process.stdin.close()
    ssh_process.stdout.close()
    ssh_process.stderr.close()


def rsync(source, dest, server, to_source=True):
    exec_remote(server, ['mkdir -r {}'.format(dest)])

    import subprocess as ps
    if to_source:
        source = "/".join(source.split("/")[:-1])
        command = ['rsync', '-r', '-azh', server['username'] + '@' + server['address'] + ':' + dest, source]
    else:
        dest = "/".join(dest.split("/")[:-1])
        command = ['rsync', '-r', '-azh', source, server['username'] + '@' + server['address'] + ':' + dest]

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
    process.stdin.close()
    process.stdout.close()
    process.stderr.close()