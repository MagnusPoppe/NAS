import os
import sys

from src.configuration import Configuration
from src.output import print_config_stats

try:
    import setproctitle

    setproctitle.setproctitle("NAS-EVOLVE")
except ImportError:
    pass
from firebase.upload import create_new_run, update_run

import src.single_net.main as ea_nas

if __name__ == "__main__":
    # Reading input arguments:
    if len(sys.argv) > 2:
        raise IOError("Program requires dataset config file.")
    if not os.path.isfile(sys.argv[1]):
        raise IOError("File {} does not exist!".format(sys.argv[1]))

    # Setting up config:
    config = Configuration.from_json(sys.argv[1])
    config.MPI = True
    config.type = "ea-nas"
    config.results.set_name(config.results.name + "_single_nas")
    run_id = create_new_run(config)
    if run_id:
        config.results_name = run_id
    print_config_stats(config)

    # Running the algorithm:
    status = "Running"
    try:
        ea_nas.run(config)
        status = "Finished"
    except KeyboardInterrupt:
        status = "Closed"
    except Exception as e:
        status = "Crashed"
        print("\nException caught in outermost loop:")
        print(e)
        raise e
    finally:
        update_run(config, status)
