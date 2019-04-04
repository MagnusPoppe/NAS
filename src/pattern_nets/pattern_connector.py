from src.helpers import randomized_index


def get_connections_between(island_c, island_n):
    # Assign which island connects to which

    # Single option on both:
    if len(island_c) == 1 and len(island_n) == 1:
        connections = [(0, 0)]

    # Single option for one of the islands:
    elif len(island_c) > 1 and len(island_n) == 1:
        connections = [(i, 0) for i in range(len(island_c))]
    elif len(island_n) > 1 and len(island_c) == 1:
        connections = [(0, i) for i in range(len(island_n))]

    # Multiple options per island:
    else:
        reverse = len(island_c) > len(island_n)
        connections = []
        selectable = randomized_index(island_n if reverse else island_c)
        for c in (range(len(island_c)) if reverse else range(len(island_n))):
            if len(selectable) == 0:
                selectable = randomized_index(island_n if reverse else island_c)
            n, selectable = selectable[0], selectable[1:]
            connections += [(c, n) if reverse else (n, c)]

    return connections
