import time
import multiprocessing as mp

def get_device_per_process(tasks:int):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    cpus = [x.name for x in local_device_protos if x.device_type == 'CPU']
    device_task_list = []

    if len(gpus) > 0:
        for gpu in gpus:
            device_task_list += [gpu] * int(tasks / len(gpus))
    else:
        for cpu in cpus:
            device_task_list += [cpu] * int(tasks / len(cpus))
    if len(device_task_list) < tasks:
        device_task_list += [gpus[0]]
    return device_task_list


def training(func, population: list, epochs: int, batch_size: int, classes:int):
    started = time.time()
    print("--> Running training for {} epochs on {} models ".format(epochs, len(population)), end="", flush=True)
    parameters = []

    devices = get_device_per_process(len(population))

    for i, individ in enumerate(population):
        individ.save_model()
        parameters += [(individ.get_store(), epochs, batch_size, classes, devices[i])]

    fitness = func(parameters[0])
    # pool = mp.Pool(processes=2)
    # result = pool.map(func, parameters)
    # pool.close()#
    # pool.join()

    print("(elapsed time: {})".format(time.time() - started))

def evaluation(func, population: list):
    print("--> Evaluating {} models".format(len(population)), end="", flush=True)
    started = time.time()
    for individ in population:
        individ.fitness = func(individ.keras_operation, individ.get_store())
    print("(elapsed time: {})".format(time.time() - started))
