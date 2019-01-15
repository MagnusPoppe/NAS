from src.buildingblocks.module import Module


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
        if last in first.next:
            first.next.remove(last)
        if first in last.prev:
            last.prev.remove(first)
    module.logs += ["Insert-between mutation" if between else "Insert mutation"]
    module.logs[-1] += " for {} between {} and {}".format(op, first, last)
    return module


def safety_insert(first, last, module) -> tuple:
    if not first in module.children or not last in module.children:
        raise Exception(
            "Tried to insert nodes between two nodes that were not part of module.")
    if _is_before(first, last):
        temp = last
        last = first
        first = temp
    return first, last


def remove(module, op) -> Module:
    """ Removes the selected Operation from the module 
        and fills the void by connecting the surrounding nodes. 
    """
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
        module.logs += [
            "Remove fully connected mutation for {} with  #prev={} #next={}".format(op, prevs, nexts)]

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
    """ Connects two nodes together within the module """"
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
