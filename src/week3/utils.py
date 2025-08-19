import random
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def shuffled(iterable: Sequence[T]) -> list[T]:
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


def argmin_random_tie(seq: Sequence[T], key: Callable[[T], float]) -> T:
    """Return a minimum element of seq based on the key value; break ties at random."""
    return min(shuffled(seq), key=key)


def argmax_random_tie(seq: Sequence[T], key: Callable[[T], float]) -> T:
    """Return a maximum element of seq; break ties at random."""
    return max(shuffled(seq), key=key)


def probability(p: float) -> bool:
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)


def define_init(city_list: list[str], start_point: str) -> list[str]:
    init_citylist = city_list[:]
    init_citylist.remove(start_point)
    init_citylist.insert(0, start_point)
    return init_citylist
