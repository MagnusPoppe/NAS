import os
import sys
import time

try:
    import setproctitle

    setproctitle.setproctitle("NAS-EVOLVE")
except ImportError:
    pass

from src.configuration import Configuration
from src.output import print_config_stats
from src.pattern_nets.main import main as evolve

print("\n\nEvolving architecture")
start_time = time.time()

if len(sys.argv) < 2:
    raise IOError("Program requires dataset config file.")
if not os.path.isfile(sys.argv[1]):
    raise IOError("File {} does not exist!".format(sys.argv[1]))

config = Configuration.from_json(sys.argv[1])
config.type = "PatternNets"
config.dataset_name = config.target_dataset.dataset_name
config.dataset_file_name = config.target_dataset.dataset_file_name
config.dataset_file_path = config.target_dataset.dataset_file_path

print_config_stats(config)
evolve(config=config)
print("\n\nTraining complete. Total runtime:", time.time() - start_time)
