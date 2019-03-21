import os
import sys
import time

from src.configuration import Configuration
from src.ea_nas.evolutionary_operations.selection import tournament
from src.pattern_nets_main import main as evolve

print("\n\nEvolving architecture")
start_time = time.time()

if len(sys.argv) > 2:
    raise IOError("Program requires dataset config file.")
if not os.path.isfile(sys.argv[1]):
    raise IOError("File {} does not exist!".format(sys.argv[1]))

config = Configuration.from_json(sys.argv[1])

evolve(selection=tournament, config=config)
print("\n\nTraining complete. Total runtime:", time.time() - start_time)
