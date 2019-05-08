from src.buildingblocks.module import Module
from src.buildingblocks.ops.convolution import Conv2D
from src.buildingblocks.ops.dense import Dense
from src.buildingblocks.ops.pooling import Pooling


def architecture_objectives() -> [callable]:
    def branching_factor(p: Module):
        def recursive_branching(current, branching, seen):
            if current in seen:
                return branching
            seen += [current]
            factor = []
            for _next in current.next:
                factor += [max(
                    branching,
                    recursive_branching(_next, branching + len(current.next) - 1, seen)
                )]
            for _prev in current.prev:
                factor += [max(
                    branching,
                    recursive_branching(_prev, branching - len(current.next) - 1, seen)
                )]
            return max(factor)

        firsts = p.find_firsts()
        seen = []
        return max(recursive_branching(first, 0, seen) for first in firsts)

    def convolution_count(p: Module):
        return len([child for child in p.children if isinstance(child, Conv2D)])

    def dense_count(p: Module):
        return len([child for child in p.children if isinstance(child, Dense)])

    def conv_pooling_factor(p: Module):
        return len([
            child
            for child in p.children
            if isinstance(child, Conv2D) and any([isinstance(n, Pooling) for n in child.next])
        ])

    def double_pooling(p: Module):
        return len([
            child
            for child in p.children
            if isinstance(child, Pooling) and any([isinstance(n, Pooling) for n in child.next])
        ])

    def overall_size(p: Module):
        return len(p.children)

    return [
        branching_factor,  # Minimize
        dense_count,  # Minimize
        double_pooling,  # Minimize
        convolution_count,  # Maximize
        conv_pooling_factor,  # Maximize
        overall_size,  # Minimize
    ]


def architecture_domination_operator(objectives: [callable]) -> callable:
    operators = objectives

    def inner(p, q) -> bool:
        """
        This represents the crowded comparison operator p dominates q
        if BOTH conditions are met:
            1. p is no worse than q in all objectives
            2. p is strictly better than q on at least one objective
        """
        comparison = [
            operators[0](q) - operators[0](p),
            operators[1](p) - operators[1](q),
            operators[3](p) - operators[3](q),
            operators[2](q) - operators[2](p),
            operators[4](q) - operators[4](p),
            operators[5](p) - operators[5](q),
        ]

        # "Strictly better" and "no worse"
        return any(c > 0 for c in comparison) and all(c >= 0 for c in comparison)

    return inner
