import os
import pickle

os.environ['EA_NAS_UPLOAD_TO_FIREBASE'] = '1'
from datasets import cifar10
from firebase.upload import create_new_run, update_status, update_run

import src.main as ea_nas

def job_start_callback(individ, config, _):
    with open(individ.get_relative_module_save_path(config) + "/genotype.obj", "wb") as f:
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

    config_file = sys.argv[1]
    with open(file=config_file, mode="r") as js:
        config = json.load(js)

    config['input'] = tuple(config['input'])
    run_id = create_new_run(config)
    if run_id:
        config['run id'] = run_id

    status = "Running"
    try:
        ea_nas.run(config, cifar10, job_start_callback, job_end_callback)
        status = "Finished"
    except KeyboardInterrupt:
        status = "SIGTERM"
    except Exception as e:
        status = "Crashed"
        raise e
    finally:
        update_run(config, status)