import os
import pickle

from src.configuration import Configuration

try:
    import setproctitle
    setproctitle.setproctitle("EA-NAS-EVOLVE")
except ImportError: pass
from src.training import cifar10
from firebase.upload import create_new_run, update_run

import src.ea_nas.main as ea_nas

def job_start_callback(individ, config, _):
    with open(individ.relative_save_path(config) + "/genotype.obj", "wb") as f:
        pickle.dump(individ, f)

def job_end_callback(manager, args, results):
    individ, config, server = args
    res = json.loads(results)
    individ.fitness += res['accuracy']
    individ.loss += res['loss']
    individ.validation_fitness += res['validation accuracy']
    individ.validation_loss += res['validation loss']
    individ.evaluation[res['eval']['epoch']] = res['eval']['accuracy']
    individ.saved_model = res['model']
    individ.model_image_path = res['image']

if __name__ == '__main__':
    import sys, json
    if len(sys.argv) > 2:
        raise IOError("Program requires dataset config file.")
    if not os.path.isfile(sys.argv[1]):
        raise IOError("File {} does not exist!".format(sys.argv[1]))

    config = Configuration.from_json(sys.argv[1])
    config.type = "ea-nas"
    run_id = create_new_run(config)
    if run_id:
        config.results_name = run_id
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
