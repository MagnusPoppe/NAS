import asyncio

import os
print(os.getcwd())

class ServerTask():

    def __init__(self, program, arguments):
        self.program = program
        self.arguments = arguments

task_queue = asyncio.Queue(maxsize=2)
queueing_futures = []


def queue_task(task):
    global queueing_futures, task_queue
    future = asyncio.ensure_future(
        task_queue.put(task),
        loop=asyncio.get_event_loop()
    )
    queueing_futures += [future]


async def consume(queue: asyncio.Queue):
    # Book-keeping
    futures = []
    i = 0

    while True:
        task = await queue.get()
        if not task:
            break

        future = asyncio.ensure_future(
            asyncio.create_subprocess_exec(
                task.program,
                *task.arguments,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        )
        futures += [future]
        queue.task_done()
        i += 1

    # Gathering results:
    results = await asyncio.gather(*futures)
    for i in range(len(results)):
        stdout, stderr = await results[i].communicate()
        results[i] = stdout.decode(), stderr.decode()
    return results

consumer_future = asyncio.ensure_future(consume(task_queue))

def finish():
    global consumer_future, task_queue
    queue_task(None)
    asyncio.get_event_loop().run_until_complete(
        asyncio.gather(*[f for f in queueing_futures if not f.done()])
    )
    results = asyncio.get_event_loop().run_until_complete(consumer_future)
    print(", ".join([res[0].strip("\n") for res in results]))

if __name__ == '__main__':
    for i in range(100000):
        queue_task(ServerTask("python", ["-u", "/work/lhome/magnuspw/ea-nas/src/distributed_processing/tester.py", f"{i}", "10"]))

    # waiting for queueing
    finish()
