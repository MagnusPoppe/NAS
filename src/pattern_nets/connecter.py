import numpy as np


def randomized_index(li: [], index_size: int = 0) -> np.array:
    index_size = len(li) if index_size == 0 else index_size
    draw = np.arange(len(li))
    np.random.shuffle(draw)
    return draw[:index_size]

def find_islands(pattern):
    """ Looks for completely separate children,
        Where there are no connections inbetween
    """
    def find_members(child, seen):
        if child in seen:
            return []
        members = [child]
        seen += [child]
        for p in child.prev:
            members += find_members(p, seen)
        for n in child.next:
            members += find_members(n, seen)
        return members

    islands = []
    seen = []
    for child in pattern.children:
        if child in seen:
            continue
        members = find_members(child, [])
        islands += [members]
        seen += members
    return islands


def island_join(patterns):
    def find_ends(island):
        def traverse(c, seen):
            starts = []
            ends = []
            if c in seen: return starts, ends
            seen += [c]
            if not c.next: ends += [c]
            if not c.prev: starts += [c]
            for n in c.next:
                n_starts, n_ends = traverse(n, seen)
                starts += n_starts
                ends += n_ends
            for p in c.prev:
                p_starts, p_ends = traverse(p, seen)
                starts += p_starts
                ends += p_ends
            return starts, ends

        all_starts = []
        all_ends = []
        for x in island:
            starts, ends = traverse(x, [])
            all_starts += starts
            all_ends += ends
        return list(set(all_starts)), list(set(all_ends))

    # Assign island connections:
    islands = [find_islands(pattern) for pattern in patterns]
    for i in range(1, len(patterns)):
        c_island = islands[i-1]
        n_island = islands[i]

        # Assign which island connects to which
        if len(c_island) == 1 and len(n_island) == 1:
            connections = [(0, 0)]
        elif len(c_island) > 1 and len(n_island) == 1:
            connections = [(i, 0) for i in range(len(c_island))]
        elif len(n_island) > 1 and len(c_island) == 1:
            connections = [(0, i) for i in range(len(n_island))]
        else:
            reverse = len(c_island) > len(n_island)
            connections = []
            selectable = randomized_index(n_island if not reverse else c_island)
            for c in (range(len(c_island)) if not reverse else range(len(n_island))):
                if len(selectable) == 0:
                    selectable = randomized_index(n_island if not reverse else c_island)
                n, selectable = selectable[0], selectable[1:]
                connections += [(c, n) if not reverse else (n, c)]

        # Connect up the different patterns:
        ops = []
        for c, n in connections:
            _, c_ends = find_ends(c_island[c])
            n_ends, _ = find_ends(n_island[n])
            for c_end in c_ends:
                for n_end in n_ends:
                    n_end.next += [c_end] if c_end not in n_end.next else []
                    c_end.prev += [n_end] if n_end not in c_end.prev else []
                    ops += [c_end, n_end]
