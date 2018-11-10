from copy import deepcopy
from operator import attrgetter

from src.evolutionary_operations.mutation_for_operators import is2D, is1D
from src.evolutionary_operations.mutation_operators import _is_before, insert
from src.buildingblocks.base import Base
from src.buildingblocks.module import Module
from src.jobs import pre_trainer as pretrain, scheduler


def get_insertion_points_after(module: Module, target: Module) -> (Base, Base):
    insertion_after = []
    last = module.find_last()[0]
    target_first = target.find_first()
    for child in module.children:
        if ((is2D(target_first) and is2D(child)) or is1D(target_first)) and child != last:
            insertion_after += [child]
    return insertion_after


def get_insertion_points_before(module: Module, target: Module, after: Base) -> (Base, Base):
    insertion_before = []
    first = module.find_first()
    target_last = target.find_last()[0]
    for child in module.children:
        if _is_before(child, after):
            if child != first and child != after and (is2D(target_last) or (is1D(target_last) and is1D(child))):
                insertion_before += [child]
    return insertion_before


def sub_module_insert(module: Module, target_module: Module, config: dict) -> Module:
    # Copying both modules to avoid destructive changes:
    target = deepcopy(target_module)

    # Finding actionspace
    action_space = []
    for start_node in get_insertion_points_after(module, target):
        for end_node in get_insertion_points_before(module, target, start_node):
            action_space += [(start_node, end_node)]

    new = []
    for predecessor_after, predecessor_before in action_space:
        # Insert module non-destructive
        mutated = deepcopy(module)  # type: module
        target = deepcopy(target_module)

        # Finding nodes in new copy:
        before, after = None, None
        for child in mutated.children:
            if child.ID == predecessor_before.ID:
                before = child
            elif child.ID == predecessor_after.ID:
                after = child

            if before and after: break

        if not before or not after:
            continue
            # raise UnboundLocalError(
            # "Insert will fail when either before ({}) or after ({}) is None".format(before, after)
            # )

        # Transferring weights and assembles module:
        mutated = insert(mutated, after, before, target)
        new += [mutated]

    if new:
        new_config = deepcopy(config)
        new_config['epochs'] = 0
        scheduler.queue_all(new, new_config, priority=True)
        new.sort(key=lambda x: x.fitness[-1])
        best = new[-1]
        print("        - Accuracy of new ({}) vs old ({})".format(best.fitness[-1], module.fitness[-1]))
        return best
    else:
        print("        - No new solutions found...")
        return None
