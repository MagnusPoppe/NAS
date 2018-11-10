import json
import pickle
import sys
import time

job = int(sys.argv[1])
mutations = int(sys.argv[2])
server = sys.argv[3]

start = time.time()
# obj = Module()
time.sleep(4)
output = {"job": job, "time elapsed": time.time() - start, "server": server}
print(json.dumps(output))
