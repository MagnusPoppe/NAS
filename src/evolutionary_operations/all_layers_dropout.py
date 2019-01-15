from copy import deepcopy

from src.buildingblocks.module import Module
from src.buildingblocks.ops.dense import Dropout
from src.evolutionary_operations.mutation_operators import insert


def apply_dropout_to_all_layers(module : Module):
    new_module = deepcopy(module)
    queue = [module.find_first()]

    while queue:
        current = queue.pop(0)

        nexts = current.next
        for _next in current.next:
            if not isinstance(_next, Dropout):
                new_module = insert(new_module, current, _next, Dropout(), between=True)
        queue += nexts


