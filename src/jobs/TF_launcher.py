import pickle
import sys

import os
import multiprocessing as mp
import time

from src.configuration import Configuration


def module_from_file(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def try_save_model(model, model_path, identity):
    """ Saving models will sometimes run out of workers.
        I cant control how others use the servers, so
        just retry up to 3 times and give up if not able
        to save.
    """
    from tensorflow import keras
    saved = False
    tries = 0
    while not saved:
        try:
            keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)
            saved = True
        except OSError:
            print("   - Failed to save model, retrying...")
            tries += 1
            if tries == 3:
                print(f"   - Failed to save model for {identity}... Training data lost.")
                return None
    return model_path


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


def unpack_arguments_and_run(args):
    timings = {'Training': 0.0, 'WriteToDisk': 0.0}
    # Unpacking arguments:
    individ_bytes, config_str, epochs, server_id, device_id, job_id = args
    individ = pickle.loads(individ_bytes)
    config = pickle.loads(config_str)

    # Finding save directory and saving genotype:
    start = time.time()
    savepath = individ.absolute_save_path(config)
    with open(os.path.join(savepath, "genotype.obj"), "wb") as f_ptr:
        pickle.dump(individ, f_ptr)
    timings['WriteToDisk'] += time.time() - start

    # Running training:
    start = time.time()
    training_module = module_from_file('cifar10', config.training_loop_path)
    model, training_history, report = training_module.main(
        individ,
        epochs,
        config,
        config.servers[server_id].devices[device_id]
    )
    timings['Training'] += time.time() - start

    # Creating results:
    start = time.time()
    model_path = os.path.join(savepath, "model.h5")
    image_path = os.path.join(savepath, individ.ID + ".png")
    model_path = try_save_model(model, model_path, individ.ID)

    try:
        from tensorflow import keras
        keras.utils.plot_model(model, to_file=image_path)
    except Exception: pass
    timings['WriteToDisk'] += time.time() - start

    print("=", end="", flush=True)
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
        "report": report,
        "timer": timings
    }


def run_jobs(population, server_id, config, verbose=True):
    start = time.time()
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
    timer = {"Training": [], "WriteToDisk": []}
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
        timer['Training'] += [res['timer']['Training']]
        timer['WriteToDisk'] += [res['timer']['WriteToDisk']]

    if verbose:
        print("--> Finished Training. Time usage:")
        print(f"    - Average Training:      {sum(timer['Training']) / len(timer['Training']):.2} sec")
        print(f"    - Average Write to disk: {sum(timer['WriteToDisk']) / len(timer['WriteToDisk']):.2} sec")
        print(f"    - Total Training:        {sum(timer['Training']):.2} sec")
        print(f"    - Total Write to disk:   {sum(timer['WriteToDisk']):.2} sec")
        print(f"    - Total:                 {time.time() - start:.2} sec")
    return population
