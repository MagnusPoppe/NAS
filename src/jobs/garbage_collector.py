import os
from src.buildingblocks.module import Module


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def collect_garbage(delete_subset: [Module], rest_population: [Module], config: dict):
    """ This function removes the "model.h5" of unused networks.
        The subset sent to this function loses its models.
        Running a simulation will create a lot of garbage networks that
        take up too much disk space.

        NOTE: Does not delete models in use. Models needed for transfer learning
        are kept on disk.
    """
    def find_predecessor_models(individ):
        collection = [individ]
        if individ.predecessor and os.path.isfile(get_model_path(individ.predecessor)):
            collection += find_predecessor_models(individ.predecessor)
        return collection

    def get_model_path(individ):
        model_path = os.path.join(individ.absolute_save_path(config), "model.h5")
        if os.path.isfile(model_path):
            return model_path
        return None

    def delete_model(individ: Module) -> bytes:
        path = get_model_path(individ)
        if path:
            size = os.stat(path).st_size
            # os.remove(path)
            return size
        return 0

    deletable = []  # type: [Module]
    for individ in delete_subset:
        for predecessor in find_predecessor_models(individ):
            if not any([
                True for x in rest_population
                if x.predecessor and predecessor.ID == x.predecessor.ID
            ]):
                deletable += [predecessor]

    storage_space_saved = 0
    for individ in deletable:
        storage_space_saved += delete_model(individ)
    print(f"--> Removed models worth {convert_bytes(storage_space_saved)}")


# if __name__ == '__main__':
#     import json, pickle
#     with open("./datasets/cifar-10-mp.json", "r") as f:
#         config = json.load(f)
#
#     population = []
#     res_path = f"./results/{config['run id']}"
#     for name in os.listdir(res_path):
#         for version in os.listdir(os.path.join(res_path, name)):
#             genotype_file = os.path.join(res_path, name, version, "genotype.obj")
#             if os.path.isfile(genotype_file):
#                 with open(genotype_file, "rb") as f:
#                     population += [pickle.load(f)]
#
#     subset, population= population[:10], population[10:]
#     collect_garbage(subset, population, config)
