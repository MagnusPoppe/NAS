import json
import multiprocessing as mp
import pickle
import time


def callback(results):
    global job_statuses
    # results = json.loads(res)
    # TODO: TEMP:
    for result in results:
        for line in result:
            js = json.loads(line)
            print(js)
            # job_id, mutation, server = result
            job_statuses[js['job']] = 0
        # print(server)


def docker_run(args):
    import subprocess as ps
    script = "/src/docker_sub_proc.py"

    program_arguments = ["docker", "run", "dsp", "python", script]
    program_arguments += [str(arg) for arg in args]
    process = ps.Popen(
        args=program_arguments,
        stdout=ps.PIPE,
        stderr=ps.PIPE,
        universal_newlines=True,
        bufsize=0
    )
    # Wait for process to finish
    while (process.poll() == None):
        time.sleep(0.1)
    if process.returncode == 0:
        output = process.stdout.readlines()
        # for i in range(len(output)):
            # output[i] = json.loads(output[i])
        return output

with open("../datasets/cifar10-config.json", "r") as f:
    config = json.load(f)

jobs = [
    (1, 2, json.dumps(config['servers'][0])),
    (2, 1, json.dumps(config['servers'][0]))
]

job_statuses = {job: 1 for job, _, _ in jobs}
processes = []

start = time.time()
pool = mp.Pool(4)
tasks = []
tasks += [pool.map_async(
    func=docker_run,
    iterable=jobs,
    callback=callback
)]

tasks += [pool.map_async(
    func=docker_run,
    iterable=jobs,
    callback=callback
)]
for map in tasks:
    res = map.get()
    print(res)
print(job_statuses)

assert time.time() - start < 9