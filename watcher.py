# watcher
import time
import sys
import os

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

def status_check(buffer, new_lines):
    if "[" == new_lines[-1][0]:
        text = buffer + new_lines
        servers = []
        i = 0
        for line in reversed(text):
            i += 1
            if "[" == line[0]:
                server = line.strip("[").split("]")[0]
                servers = servers + [server] if not server in servers else servers
            elif "--> Starting MPI Pool executor" in line:
                occupancy_str = line.strip(" ").split(".")[1].split(" ")
                jobs = int(occupancy_str[1])
                num_servers = int(occupancy_str[5])
                return "Status: {}/{} started. Occupancy at {} %".format(
                    i, jobs, float(len(servers)) / (num_servers-1)
                )
    else:
        return None


def read_on_interval(file, interval):
    os.system("clear")
    buffer = []
    status = None
    prev_status = None
    while True:
        with open(file, "r") as f:
            new_lines = f.readlines()[len(buffer):]

            if new_lines: status = status_check(buffer, new_lines)

            if status and prev_status:
                sys.stdout.write(CURSOR_UP_ONE + ERASE_LINE)
                prev_status = None

            for line in new_lines:
                sys.stdout.write(line)

            if status:
                sys.stdout.write(status)
                prev_status = status
                status = None
            buffer += new_lines
        time.sleep(interval)


file = sys.argv[1]
timer = int(sys.argv[2]) if len(sys.argv) > 2 else 2

try:
    read_on_interval(file, timer)
except KeyboardInterrupt:
    exit(0)
