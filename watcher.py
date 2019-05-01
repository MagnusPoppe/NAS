# watcher
import time
import sys
import os

def read_on_interval(file, interval):
    os.system("clear")
    buffer = []
    while True:
        with open(file, "r") as f:
            new_lines = f.readlines()

            # Reset condition:
            if len(new_lines) < len(buffer):
                buffer = []

            new_lines = new_lines[len(buffer):]
            for line in new_lines:
               sys.stdout.write(line)
            buffer += new_lines
        time.sleep(interval)


file = sys.argv[1]
timer = int(sys.argv[2]) if len(sys.argv) > 2 else 2

try:
    read_on_interval(file, timer)
except KeyboardInterrupt:
    exit(0)