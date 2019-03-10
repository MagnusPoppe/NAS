import pickle
import json
import sys

import os
import multiprocessing as mp

from src.configuration import Configuration


def module_from_file(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unpack_arguments_and_run(args):
    from tensorflow import keras
    # Unpacking arguments:
    individ_bytes, config_str, epochs, server_id, device_id, job_id = args
    individ = pickle.loads(individ_bytes)
    config = pickle.loads(config_str)

    # Finding save directory and saving genotype:
    savepath = individ.absolute_save_path(config)
    with open(os.path.join(savepath, "genotype.obj"), "wb") as f_ptr:
        pickle.dump(individ, f_ptr)

    # Running training:
    training_module = module_from_file('cifar10', config.training_loop_path)
    model, training_history, report = training_module.main(
        individ,
        epochs,
        config,
        config.servers[server_id].devices[device_id]
    )

    # Creating results:
    model_path = os.path.join(savepath, "model.h5")
    image_path = os.path.join(savepath, individ.ID + ".png")
    keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)

    # save_model_image(model, image_path)
    print("=", end="")
    return {
        "job": job_id,
        "image": image_path,
        "model": model_path,
        "epochs": epochs,
        "accuracy": training_history["acc"],
        "validation accuracy": training_history["val_acc"],
        "test accuracy": report['weighted avg']['precision'],
        "loss": training_history["loss"],
        "validation loss": training_history["val_loss"],
        "report": report
    }


def pack_args(population, server_id, config: Configuration):
    """ Compiles a list of arguments for parallel training """
    config_str = pickle.dumps(config)

    # Each server gets a portion of the jobs:
    server_job_args = [[] for _ in range(len(config.servers[server_id].devices))]

    # Calculating size of each job:
    sized = []
    for ind in population:
        epochs = int(ind.number_of_operations() * config.epochs_per_layer)
        sized += [epochs if epochs > 0 else 1]
    total_epochs = sum(sized)

    # Create balanced workloads for each process by estimate:
    for i in range(len(population)):
        dev_id = i % len(config.servers[server_id].devices)
        job_id = sized.index(max(sized))

        # Compiling list of arguments:
        server_job_args[dev_id] += [(
            pickle.dumps(population[job_id]),
            config_str,
            int(sized[job_id]),
            server_id,
            dev_id,
            job_id
        )]
        sized[job_id] = - sys.maxsize

    return server_job_args, total_epochs


def run_jobs(population, server_id, config):
    server_args, epochs = pack_args(population, server_id, config)
    print(f"--> Running {epochs} epoch(s) of training for {len(population)} phenotype(s)")
    print("  |", end="")

    # Spawning jobs:
    pools = []
    pool_res = []
    for id, device in enumerate(config.servers[server_id].devices):
        pool = mp.Pool(processes=device.concurrency)
        pool_res += [
            pool.map_async(unpack_arguments_and_run, server_args[id])
        ]
        pools += [pool]

    # Awaiting results:
    results = []
    for pool, res in zip(pools, pool_res):
        pool.close()
        results += res.get()
    print("|")

    # Applying results:
    results.sort(key=lambda x: x["job"])
    for individ, res in zip(population, results):
        individ.fitness += res['accuracy']
        individ.loss += res['loss']
        individ.validation_fitness += res['validation accuracy']
        individ.validation_loss += res['validation loss']
        individ.evaluation[res['epochs']] = res['test accuracy']
        individ.epochs_trained += res['epochs']
        individ.report[individ.epochs_trained] = res['report']
        individ.saved_model = res['model']
        individ.model_image_path = res['image']
    return population
