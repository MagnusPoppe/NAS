import random
from copy import deepcopy

from helpers import random_sample, operators1D
from frameworks.keras_decoder import assemble
from modules.module import Module


def _generate_votes(weights: list) -> list:
    votes = []
    for operation, weight in weights:
        votes += [operation] * weight
    return votes


OPERATOR_WEIGHTS = [
    ("append", 2),
    ("connect", 5),
    ("insert", 10),
    ("insert-between", 10),
    ("remove", 30),
    ("identity", 20),
]

votes = _generate_votes(OPERATOR_WEIGHTS)


def select_operator(module):
    """ Selects what mutation operator to use """
    if len(module.children) <= 3:
        return "append"
    return random_sample(votes)


def select_addon(operators, modules):
    if not modules or random.uniform(0, 1) > 0.5:
        return random_sample(operators)()
    else:
        return random_sample(modules)


def apply_mutation_operator(module, operator, modules, operators):
    if operator == "append":
        module = append(module, select_addon(operators, modules))
    elif operator == "remove":
        module = remove(module, random_sample(module.children))
    elif operator == "insert" or operator == "insert-between":
        possibilities = list(range(len(module.children)))
        module = insert(
            module=module,
            op=select_addon(operators, modules),
            first=module.children[possibilities.pop(random.randint(0, len(possibilities) - 1))],
            last=module.children[possibilities.pop(random.randint(0, len(possibilities) - 1))],
            between=operator == "insert-between"
        )
    elif operator == "connect":
        possibilities = list(range(len(module.children)))
        module = connect(
            module=module,
            first=module.children[possibilities.pop(random.randint(0, len(possibilities) - 1))],
            last=module.children[possibilities.pop(random.randint(0, len(possibilities) - 1))]
        )
    # Else: operator == "identity": do nothing...

    return module


def mutate(module: Module, in_shape: tuple, classes: int, modules: list = None, compilation: bool = True) -> Module:
    mutated = deepcopy(module)
    mutated = apply_mutation_operator(mutated, select_operator(mutated), modules, operators1D)

    # Compiles keras model from module (not done for multiple mutations):
    if compilation:
        try:
            if random.uniform(0, 1) < 0.2 or not module.predecessor:
                mutated.keras_tensor = assemble(mutated, in_shape, classes, is_root=True)
            else:
                print("--> {} got its weights transferred from predecessor".format(module.ID))
                mutated = transfer_predecessor_weights(mutated, in_shape, classes)
        except ValueError as e:
            print("--> Transfer Predecessor Weights: I got an invalid module as input...")
            return None  # Invalid network!
    return mutated



def transfer_predecessor_weights(module: Module, in_shape: tuple, classes: int) -> Module:
    """ Transfers the weights from one module to its successor. This requires
        compatibility with changes made to the successor. Some types of changes
        will change weight matrix sizes.

        These problems raise ValueError. (anomaly)
         - ValueError is ignored.
         - Module / Operation keeps its random initialized weights.
    """
    predecessor = module.predecessor
    module.keras_tensor = assemble(module, in_shape, classes)

    for predecessor_child in predecessor.children:
        for child in module.children:
            if child.ID == predecessor_child.ID:
                try:
                    child.keras_operation.set_weights(predecessor_child.keras_operation.get_weights())
                except ValueError: break
    return module


def append(module, op) -> Module:
    if len(module.children) > 0:
        last = module.find_last()[0]
        op.prev += [last]
        last.next += [op]
    module.children += [op]
    module.logs += ["Append mutation for {}".format(op)]
    return module


def insert(module, first, last, op, between=False) -> Module:
    # Check that "first" and "last" is in the list of children
    first, last = safety_insert(first, last, module)

    # Add new operation to list of operations within module:
    module.children.append(op)

    # Fully connect first with op and op with last
    op.prev += [first]
    op.next += [last]
    first.next += [op]
    last.prev += [op]

    if between:
        if last in first.next: first.next.remove(last)
        if first in last.prev: last.prev.remove(first)
    module.logs += ["Insert-between mutation" if between else "Insert mutation"]
    module.logs[-1] += " for {} between {} and {}".format(op, first, last)
    return module


def safety_insert(first, last, module) -> tuple:
    if not first in module.children or not last in module.children:
        raise Exception("Tried to insert nodes between two nodes that were not part of module.")
    if _is_before(first, last):
        temp = last
        last = first
        first = temp
    return first, last


def remove(module, op) -> Module:
    if op.next and op.prev:
        module.children.remove(op)
        prevs, nexts = len(op.prev), len(op.next)
        # Connecting ops previous nodes to its next nodes, bypassing it self:
        for prev_op in op.prev:
            for next_op in op.next:
                if next_op not in prev_op.next:
                    prev_op.next += [next_op]
                if prev_op not in next_op.prev:
                    next_op.prev += [prev_op]

        # Removing ties between op and its previous and next nodes:
        for prev_op in op.prev:
            prev_op.next.remove(op)
        for next_op in op.next:
            next_op.prev.remove(op)
        op.prev = []
        op.next = []
        module.logs += ["Remove fully connected mutation for {} with  #prev={} #next={}".format(op, prevs, nexts)]

    elif len(op.next) == 1:
        # Can only delete first node when it has a single connection forwards.
        module.children.remove(op)
        op.next[0].prev.remove(op)
        op.next = []
        module.logs += ["Remove first mutation for {}".format(op)]
    elif len(op.prev) == 1:
        # Can only delete last node when it has a single connection backwards.
        module.children.remove(op)
        op.prev[0].next.remove(op)
        op.prev = []
        module.logs += ["Remove last mutation for {}".format(op)]


    return module


def connect(module, first, last) -> Module:
    first, last = safety_insert(first, last, module)
    if first in last.prev or last in first.next:
        return module
    else:
        first.next += [last]
        last.prev += [first]
    module.logs += ["Connect mutation between {} and {}".format(first, last)]
    return module


def _is_before(node, target) -> bool:
    """ :returns: True if "target" is before "node" in directed acyclic graph """
    if node == target:
        return True
    elif node.prev:
        return any([_is_before(prev, target) for prev in node.prev])
    else:
        return False
