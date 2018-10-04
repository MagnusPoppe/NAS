from modules.module import Module


def rank_children(module: Module) -> Module:
    """ Ranks all children of module in breadth first order. This
        makes sorting all nodes after Keras operations possible.
    """
    for node in module.children:
        node.rank = -1

    queue = [module.find_first()]
    rank = 0
    while queue:
        node = queue.pop(0) # type: Base

        # Should wait to queue next nodes if one or more previous nodes are "unprocessed"
        if (not node.prev) or all(_prev.rank >= 0 for _prev in node.prev):
            queue += [_next for _next in node.next]
            node.rank = rank
            rank += 1