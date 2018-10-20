from copy import deepcopy
from operator import attrgetter

from evolutionary_operations.mutation_for_operators import is2D, is1D
from evolutionary_operations.mutation_operators import _is_before, insert
from evolutionary_operations.weight_transfer import transfer_predecessor_weights
from frameworks.keras_decoder import assemble
from modules.base import Base
from modules.module import Module


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


def sub_module_insert(module: Module, target_module: Module, in_shape: tuple, classes: int, train) -> Module:
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
            raise UnboundLocalError("Insert will fail when either before ({}) or after ({}) is None".format(before, after))

        # Transferring weights and assembles module:
        mutated = insert(mutated, after, before, target)
        mutated.keras_operation = assemble(mutated, in_shape, classes)
        mutated = transfer_predecessor_weights(mutated, in_shape, classes)

        # Transfer weights of target module to sub module
        sub_modules = [m for m in mutated.children if isinstance(m, Module) and m.name == target.name]
        for sub_module in sub_modules:
            transfer_predecessor_weights(sub_module, in_shape, classes, predecessor=target_module)
        new += [mutated]

    if new:
        train(new, 1, 1024, prefix="        - ")
        new.sort(key=attrgetter("fitness"))
        best = new[-1]
        print("        - Accuracy of new ({}) vs old ({})".format(best.fitness, module.fitness))
        return best
    else:
        print("        - No new solutions found...")
        return module
