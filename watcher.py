# watcher
import time
import sys
import os

def count_jobs_started(start, lines, ending):
    i = 0
    for line in lines[start:]:
        if ending in line:
           break
        i += 1
    return i

def status_check(status, buffer, new_lines):
    def make_text(start, lines):
        numbers = [x for x in lines[start].split() if x.isnumeric()]
        jobs, servers = int(numbers[0]), int(numbers[1])
        return "Training status: {}/{} started training".format(
            count_jobs_started(start, lines, "--> Entire population trained. "),
            servers
        )

    if any("--> Entire population trained" in line for line in new_lines):
        status = ""
    elif any("--> Starting MPI Pool executor" in line for line in new_lines):
        start_line = [i for i, line in enumerate(new_lines) if "--> Starting MPI Pool executor" in line][-1]
        numbers = [x for x in new_lines[start_line].split() if x.isnumeric()]
        jobs, servers = int(numbers[0]), int(numbers[1])
        status = "Training status: {}/{} started training".format(
            count_jobs_started(start_line, new_lines, "--> Entire population trained."),
            servers
        )
    elif status:
        joined = buffer + new_lines
        for i in range(len(joined) - 1, 0, -1):
            if "--> Starting MPI Pool executor" in joined[i]:
                status = make_text(i, joined)
                break
    return status


def read_on_interval(file, interval):
    os.system("clear")
    buffer = []
    status = ""
    while True:
        with open(file, "r") as f:
            new_lines = f.readlines()

            # Reset condition:
            if len(new_lines) < len(buffer):
                buffer = []

            new_lines = new_lines[len(buffer):]
            if new_lines:
                status = status_check(status, buffer, new_lines)

            if status:
                for line in new_lines:
                   sys.stdout.write(line)
                sys.stdout.write(status + "\r")

            else:
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


"""
Configuration for None:
Evolutionary algorithm parameters:
	Type:                          ea-nas
	Population size:               2
	Generations:                   10
	Number of pattern/layers used: 2 - 5
	Elitism sorting type:          Weighted scoring
Neural network training:
	Epochs:                        1 (Fixed)
	Minibatch size:                90
	Learning rate:                 0.001
	Use restarting:                False
Servers:
	Number of servers:             2
	Number of compute devices:     2
	Results save location:         /lhome/magnuspw/ea-nas/results/results/simulation01_ea_nas
	Delete unused results:         True



Evolving architecture
Starting MPI Pool executor. 2 jobs running on 2 servers
[luke01 /GPU:0]:	Training Magnhild v0
[luke01 /GPU:1]:	Training Ole-kristian v0
--> Initialization complete. Leaderboards:
SPECIMIN          ACC    VACC   IMPR   OPS   LR      1      2      3      4      5      6      7      8      9      10     MIAVG   MAAVG   WAVG   
Magnhild v0       88.3   96.5   -      3     0.001   96.9   97.5   96.7   97.2   95.4   96.4   96.3   97.0   96.4   96.2   96.6    96.6    96.6   
Ole-kristian v0   94.0   97.8   -      2     0.001   98.5   97.7   96.5   97.5   98.1   97.9   97.6   97.4   97.1   97.4   97.6    97.6    97.6   


Generation 0
--> Mutations:
    - Operation Mutation for Ole-kristian v0
Starting MPI Pool executor. 3 jobs running on 2 servers
[luke01 /GPU:0]:	Training Magnhild v0
[luke01 /GPU:1]:	Training Ole-kristian v0
[luke01 /GPU:0]:	Training Ole-kristian v2
--> Generation 0 Leaderboards:
SPECIMIN          ACC    VACC   IMPR   OPS   LR      1      2      3      4      5      6      7      8      9      10     MIAVG   MAAVG   WAVG   
Magnhild v0       96.3   98.3   -      3     0.001   99.0   98.9   97.8   97.4   98.2   98.2   98.7   98.0   98.5   97.0   98.2    98.2    98.2   
Ole-kristian v0   97.8   98.0   -      2     0.001   98.2   98.9   97.4   97.3   98.4   97.9   98.1   97.3   96.9   96.9   97.8    97.7    97.8   

--> The following individs were removed by elitism:
SPECIMIN          ACC    VACC   IMPR    OPS   LR      1      2      3      4      5      6      7      8      9      10     MIAVG   MAAVG   WAVG   
Ole-kristian v2   96.7   98.2   0.003   3     0.001   98.7   98.7   97.6   98.3   98.5   96.8   97.3   97.8   97.6   97.6   97.9    97.9    97.9   

--> Possible final solution discovered. Checking...
Starting MPI Pool executor. 2 jobs running on 2 servers
[luke01 /GPU:0]:	Training Magnhild v0
[luke01 /GPU:1]:	Training Ole-kristian v0
--> Found final solution:
SPECIMIN          ACC    VACC   IMPR   OPS   LR      1      2      3      4      5      6      7      8      9      10     MIAVG   MAAVG   WAVG   
Ole-kristian v0   98.5   98.0   -      2     0.001   98.7   97.3   97.5   99.0   98.4   98.4   98.1   96.5   98.4   97.7   98.0    98.0    98.0   
Magnhild v0       97.4   98.8   -      3     0.001   98.7   99.2   98.1   98.9   98.9   97.9   98.4   98.1   98.5   97.8   98.5    98.4    98.5   



Training complete. Total runtime: 68.82403135299683

Process finished with exit code 0"""
