import pickle
import json
import sys

import os
import multiprocessing as mp

def module_from_file(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unpack_arguments_and_run(args):
    from tensorflow import keras
    # Unpacking arguments:
    individ_bytes, config_str, server_id, job_id = args
    individ = pickle.loads(individ_bytes)
    config = json.loads(config_str)

    # Finding save directory and saving genotype:
    savepath = individ.absolute_save_path(config)
    with open(os.path.join(savepath, "genotype.obj"), "wb") as f_ptr:
        pickle.dump(individ, f_ptr)

    # Running training:
    training_module = module_from_file('cifar10', config['trainingFilepath'])
    model, training_history, after = training_module.main(individ, config, config['servers'][server_id])

    # Creating results:
    model_path = os.path.join(savepath, "model.h5")
    image_path = os.path.join(savepath, individ.ID + ".png")
    keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)
    # save_model_image(model, image_path)
    print(f"    - [{job_id}] {individ.ID} finished training")
    return {
        "job": job_id,
        "image": image_path,
        "model": model_path,
        "accuracy": training_history["acc"],
        "validation accuracy": training_history["val_acc"],
        "loss": training_history["loss"],
        "validation loss": training_history["val_loss"],
        "eval": {
            "epoch": str(len(training_history) + len(individ.fitness) - 2),
            "accuracy": after,
        }
    }

def pack_args(population, config):
    # Compiling a list of arguments for parallel training:
    config_str = json.dumps(config)

    # Each server gets a portion of the jobs:
    server_job_args = [[] for _ in range(len(config["servers"]))]

    # TODO: Pack according to size as a mini-plan...
    # for job_id, arg in enumerate(population):
    #     server_id = job_id % len(config["servers"])
    #     server_job_args[server_id] += [(pickle.dumps(arg), config_str, server_id, job_id)]

    # Create balanced workloads for each process by estimate:
    sized = [ind.number_of_operations() * config['epochs'] if config['epochs'] > 0 else 1 for ind in population]
    for i in range(len(population)):
        server_id = i % len(config['servers'])
        job_id = sized.index(max(sized))
        sized[job_id] = - sys.maxsize
        server_job_args[server_id] +=  [(pickle.dumps(population[job_id]), config_str, server_id, job_id)]
    return server_job_args

def run_jobs(population, config):
    print(f"--> Running training for {len(population)} phenotypes")
    server_args = pack_args(population, config)

    # Spawning jobs:
    pools = []
    pool_res = []
    for server_id, server in enumerate(config['servers']):
        pool = mp.Pool(processes=server['concurrency'])
        pool_res += [pool.map_async(unpack_arguments_and_run, server_args[server_id])]
        pools += [pool]

    # Awaiting results:
    results = []
    for pool, res in zip(pools, pool_res):
        pool.close()
        results += res.get()

    # Applying results:
    results.sort(key=lambda x: x["job"])
    for individ, res in zip(population, results):
        individ.fitness += res['accuracy']
        individ.loss += res['loss']
        individ.validation_fitness += res['validation accuracy']
        individ.validation_loss += res['validation loss']
        individ.evaluation[res['eval']['epoch']] = res['eval']['accuracy']
        individ.saved_model = res['model']
        individ.model_image_path = res['image']
    return population
