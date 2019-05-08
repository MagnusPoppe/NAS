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

# Configuring
print(f"Using JSON configuration {sys.argv[1]}")
print(sys.argv)
config = Configuration.from_json(sys.argv[1])
config.type = "PatternNets"
config.MPI = True
config.results.set_name(config.results.name + "_pattern_nas")
config.dataset_name = config.target_dataset.dataset_name
config.dataset_file_name = config.target_dataset.dataset_file_name
config.dataset_file_path = config.target_dataset.dataset_file_path
config.training.acceptable_scores = config.target_dataset.accepted_accuracy
config.input_format = config.target_dataset.input
config.augmentations = config.target_dataset.augmentations

print_config_stats(config)
evolve(config=config)
print("\n\nTraining complete. Total runtime:", time.time() - start_time)
