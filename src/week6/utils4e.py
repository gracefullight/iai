from __future__ import annotations

import bisect
import collections
import collections.abc
import functools
import heapq
import os.path
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import chain, combinations
from statistics import mean
from typing import Any, TypeVar, IO, TextIO, Tuple, List, Dict, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

# part1. General data structures and their functions
# ______________________________________________________________________________
# Queues: Stack, FIFOQueue, PriorityQueue
# Stack and FIFOQueue are implemented as list and collection.deque
# PriorityQueue is implemented here


TV = TypeVar("TV")


class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup.
    """

    def __init__(self, order: str = "min", f: Callable[[Any], Any] = lambda x: x):
        self.heap: list[tuple[Any, Any]] = []

        if order == "min":
            self.f = f
        elif order == "max":  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item: Any) -> None:
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items: Iterable[Any]) -> None:
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self) -> Any:
        """Pop and return the item (with min or max f(x) value)
        depending on the order.
        """
        if self.heap:
            return heapq.heappop(self.heap)[1]
        raise Exception("Trying to pop from empty PriorityQueue.")

    def __len__(self) -> int:
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key: Any) -> bool:
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key: Any) -> Any:
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present.
        """
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key: Any) -> None:
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)


# ______________________________________________________________________________
# Functions on Sequences and Iterables


def sequence(iterable: Iterable[Any]) -> Sequence[Any] | tuple[Any, ...]:
    """Converts iterable to sequence, if it is not already one."""
    return iterable if isinstance(iterable, collections.abc.Sequence) else tuple([iterable])


def remove_all(item: Any, seq: Any) -> Any:
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, "")
    if isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    return [x for x in seq if x != item]


def unique(seq: Iterable[Any]) -> list[Any]:
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))


def count(seq: Iterable[Any]) -> int:
    """Count the number of items in sequence that are interpreted as true."""
    return sum(map(bool, seq))


def multimap(items: Iterable[tuple[Any, Any]]) -> dict[Any, list[Any]]:
    """Given (key, val) pairs, return {key: [val, ....], ...}."""
    result = collections.defaultdict(list)
    for key, val in items:
        result[key].append(val)
    return dict(result)


def multimap_items(mmap: Mapping[Any, list[Any]]) -> Iterable[tuple[Any, Any]]:
    """Yield all (key, val) pairs stored in the multimap."""
    for key, vals in mmap.items():
        for val in vals:
            yield key, val


def product(numbers: Iterable[int]) -> int:
    """Return the product of the numbers, e.g. product([2, 3, 10]) == 60"""
    result = 1
    for x in numbers:
        result *= x
    return result


def first(iterable: Iterable[TV], default: Optional[TV] = None) -> Optional[TV]:
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)


def is_in(elt: Any, seq: Iterable[Any]) -> bool:
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def mode(data: Iterable[Any]) -> Any:
    """Return the most common data item. If there are ties, return any one of them."""
    [(item, count)] = collections.Counter(data).most_common(1)
    return item


def power_set(iterable: Iterable[Any]) -> list[tuple[Any, ...]]:
    """power_set([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))[1:]


def extend(s: Mapping[str, Any], var: str, val: Any) -> dict[str, Any]:
    """Copy dict s and extend it by setting var to val; return copy."""
    return {**s, var: val}


def flatten(seqs: Iterable[Iterable[Any]]) -> list[Any]:
    return [item for seq in seqs for item in seq]


# ______________________________________________________________________________
# argmin and argmax


identity: Callable[[TV], TV] = lambda x: x


def argmin_random_tie(seq: Sequence[TV], key: Callable[[TV], Any] = identity) -> TV:
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)


def argmax_random_tie(seq: Sequence[TV], key: Callable[[TV], Any] = identity) -> TV:
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return max(shuffled(seq), key=key)


def shuffled(iterable: Iterable[TV]) -> list[TV]:
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


# part2. Mathematical and Statistical util functions
# ______________________________________________________________________________


def histogram(
    values: Iterable[Any], mode: int = 0, bin_function: Callable[[Any], Any] | None = None
) -> list[tuple[Any, int]]:
    """Return a list of (value, count) pairs, summarizing the input values.
    Sorted by increasing value, or if mode=1, by decreasing count.
    If bin_function is given, map it over values first.
    """
    if bin_function:
        values = map(bin_function, values)

    bins: dict[Any, int] = {}
    for val in values:
        bins[val] = bins.get(val, 0) + 1

    if mode:
        return sorted(list(bins.items()), key=lambda x: (x[1], x[0]), reverse=True)
    return sorted(bins.items())


def element_wise_product(x: Any, y: Any) -> Any:
    if hasattr(x, "__iter__") and hasattr(y, "__iter__"):
        assert len(x) == len(y)
        return [element_wise_product(_x, _y) for _x, _y in zip(x, y, strict=False)]
    if hasattr(x, "__iter__") == hasattr(y, "__iter__"):
        return x * y
    raise Exception("Inputs must be in the same size!")


def vector_add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    """Component-wise addition for 2D grid vectors.

    This specialized version matches how GridMDP uses it: both inputs are
    2D integer tuples representing (x, y) directions or states.
    """
    return (a[0] + b[0], a[1] + b[1])


def scalar_vector_product(x: Any, y: Any) -> Any:
    """Return vector as a product of a scalar and a vector recursively."""
    return [scalar_vector_product(x, _y) for _y in y] if hasattr(y, "__iter__") else x * y


def map_vector(f: Callable[[Any], Any], x: Any) -> Any:
    """Apply function f to iterable x."""
    return [map_vector(f, _x) for _x in x] if hasattr(x, "__iter__") else list(map(f, [x]))[0]


def probability(p: float) -> bool:
    """Return true with probability p."""
    return p > random.uniform(0.0, 1.0)


def weighted_sample_with_replacement(n: int, seq: Sequence[TV], weights: Sequence[float]) -> list[TV]:
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight.
    """
    sample = weighted_sampler(seq, weights)

    return [sample() for _ in range(n)]


def weighted_sampler(seq: Sequence[TV], weights: Sequence[float]) -> Callable[[], TV]:
    """Return a random-sample function that picks from seq weighted by weights."""
    totals: list[float] = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def weighted_choice(choices: Iterable[tuple[TV, float]]) -> tuple[TV, float] | None:
    """A weighted version of random.choice"""
    # NOTE: Should be replaced by random.choices if we port to Python 3.6

    total = sum(w for _, w in choices)
    r = random.uniform(0.0, total)
    upto: float = 0.0
    for c, w in choices:
        if upto + w >= r:
            return c, w
        upto += w
    return None


def rounder(numbers: float | Iterable[float], d: int = 4) -> float | list[float]:
    """Round a single number or a sequence of numbers to d decimal places."""
    if isinstance(numbers, (int, float)):
        return round(float(numbers), d)
    return [round(float(n), d) for n in numbers]


def num_or_str(x: str) -> Union[int, float, str]:  # TODO: rename as `atom`
    """The argument is a string; convert to a number if
    possible, or strip it.
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def euclidean_distance(x: Iterable[float], y: Iterable[float]) -> float:
    return float(np.sqrt(sum((_x - _y) ** 2 for _x, _y in zip(x, y, strict=False))))


def manhattan_distance(x: Iterable[float], y: Iterable[float]) -> float:
    return float(sum(abs(_x - _y) for _x, _y in zip(x, y, strict=False)))


def hamming_distance(x: Iterable[Any], y: Iterable[Any]) -> int:
    return int(sum(_x != _y for _x, _y in zip(x, y, strict=False)))


def rms_error(x: Iterable[float], y: Iterable[float]) -> float:
    return float(np.sqrt(ms_error(x, y)))


def ms_error(x: Iterable[float], y: Iterable[float]) -> float:
    return float(mean((x - y) ** 2 for x, y in zip(x, y, strict=False)))


def mean_error(x: Iterable[float], y: Iterable[float]) -> float:
    return float(mean(abs(x - y) for x, y in zip(x, y, strict=False)))


def mean_boolean_error(x: Iterable[Any], y: Iterable[Any]) -> float:
    return float(mean(_x != _y for _x, _y in zip(x, y, strict=False)))


# part3. Neural network util functions
# ______________________________________________________________________________


def cross_entropy_loss(x: Sequence[float], y: Sequence[float]) -> float:
    """Cross entropy loss function. x and y are 1D iterable objects."""
    return float(
        (-1.0 / len(x))
        * sum(_x * float(np.log(_y)) + (1 - _x) * float(np.log(1 - _y)) for _x, _y in zip(x, y, strict=False))
    )


def mean_squared_error_loss(x: Sequence[float], y: Sequence[float]) -> float:
    """Min square loss function. x and y are 1D iterable objects."""
    return float((1.0 / len(x)) * sum((_x - _y) ** 2 for _x, _y in zip(x, y, strict=False)))


from typing import overload


@overload
def normalize(dist: Dict[Any, float]) -> Dict[Any, float]: ...


@overload
def normalize(dist: Sequence[float]) -> list[float]: ...


def normalize(dist: Union[Dict[Any, float], Sequence[float]]) -> Union[Dict[Any, float], list[float]]:
    """Multiply each number by a constant such that the sum is 1.0"""
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1  # probabilities must be between 0 and 1
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


def random_weights(min_value: float, max_value: float, num_weights: int) -> list[float]:
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]


def conv1D(x: NDArray[np.floating[Any]], k: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """1D convolution. x: input vector; K: kernel vector."""
    return cast(NDArray[np.floating[Any]], np.convolve(x, k, mode="same"))


def gaussian_kernel(size: int = 3) -> list[float]:
    return [gaussian((size - 1) / 2, 0.1, float(x)) for x in range(size)]


def gaussian_kernel_1D(size: int = 3, sigma: float = 0.5) -> list[float]:
    return [gaussian((size - 1) / 2, sigma, float(x)) for x in range(size)]


def gaussian_kernel_2D(size: int = 3, sigma: float = 0.5) -> NDArray[np.floating[Any]]:
    x, y = np.mgrid[-size // 2 + 1 : size // 2 + 1, -size // 2 + 1 : size // 2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return cast(NDArray[np.floating[Any]], g / g.sum())


def step(x: float) -> int:
    """Return activation value of x with sign function."""
    return 1 if x >= 0 else 0


def gaussian(mean: float, st_dev: float, x: float) -> float:
    """Given the mean and standard deviation of a distribution, it returns the probability of x."""
    return float(1 / (np.sqrt(2 * np.pi) * st_dev) * np.exp(-0.5 * (float(x - mean) / st_dev) ** 2))


def linear_kernel(x: NDArray[np.floating[Any]], y: Optional[NDArray[np.floating[Any]]] = None) -> NDArray[np.floating[Any]]:
    if y is None:
        y = x
    return cast(NDArray[np.floating[Any]], np.dot(x, y.T))


def polynomial_kernel(
    x: NDArray[np.floating[Any]], y: Optional[NDArray[np.floating[Any]]] = None, degree: float = 2.0
) -> NDArray[np.floating[Any]]:
    if y is None:
        y = x
    return cast(NDArray[np.floating[Any]], (1.0 + np.dot(x, y.T)) ** degree)


def rbf_kernel(
    x: NDArray[np.floating[Any]], y: Optional[NDArray[np.floating[Any]]] = None, gamma: Optional[float] = None
) -> NDArray[np.floating[Any]]:
    """Radial-basis function kernel (aka squared-exponential kernel)."""
    if y is None:
        y = x
    if gamma is None:
        gamma = 1.0 / x.shape[1]  # 1.0 / n_features
    return cast(
        NDArray[np.floating[Any]],
        np.exp(
            -gamma
            * (
                -2.0 * np.dot(x, y.T)
                + np.sum(x * x, axis=1).reshape((-1, 1))
                + np.sum(y * y, axis=1).reshape((1, -1))
            )
        ),
    )


# part4. Self defined data structures
# ______________________________________________________________________________
# Grid Functions


EAST: tuple[int, int]
NORTH: tuple[int, int]
WEST: tuple[int, int]
SOUTH: tuple[int, int]
EAST, NORTH, WEST, SOUTH = (1, 0), (0, 1), (-1, 0), (0, -1)
orientations: list[tuple[int, int]] = [EAST, NORTH, WEST, SOUTH]
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(
    heading: tuple[int, int], inc: int, headings: Sequence[tuple[int, int]] = orientations
) -> tuple[int, int]:
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading: tuple[int, int]) -> tuple[int, int]:
    return turn_heading(heading, RIGHT)


def turn_left(heading: tuple[int, int]) -> tuple[int, int]:
    return turn_heading(heading, LEFT)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """The distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return float(np.hypot((xA - xB), (yA - yB)))


def distance_squared(a: tuple[float, float], b: tuple[float, float]) -> float:
    """The square of the distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return (xA - xB) ** 2 + (yA - yB) ** 2


# ______________________________________________________________________________
# Misc Functions


class injection:
    """Dependency injection of temporary values for global functions/classes/etc.
    E.g., `with injection(DataBase=MockDataBase): ...`
    """

    def __init__(self, **kwds: Any) -> None:
        self.new: Dict[str, Any] = kwds  # type: ignore[assignment]

    def __enter__(self) -> "injection":
        self.old = {v: globals()[v] for v in self.new}
        globals().update(self.new)
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        globals().update(self.old)


def memoize(fn: Callable[..., Any], slot: Optional[str] = None, maxsize: int = 32) -> Callable[..., Any]:
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values.
    """
    if slot:

        def memoized_fn(obj: Any, *args: Any) -> Any:
            if hasattr(obj, slot):
                return getattr(obj, slot)
            val = fn(obj, *args)
            setattr(obj, slot, val)
            return val
    else:

        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args: Any) -> Any:
            return fn(*args)

    return memoized_fn


def name(obj: Any) -> str:
    """Try to find some reasonable name for the object."""
    cls = getattr(obj, "__class__", None)
    return (
        getattr(obj, "name", "")
        or getattr(obj, "__name__", "")
        or (getattr(cls, "__name__", "") if cls is not None else "")
        or str(obj)
    )


def isnumber(x: Any) -> bool:
    """Is x a number?"""
    return hasattr(x, "__int__")


def issequence(x: Any) -> bool:
    """Is x a sequence?"""
    return isinstance(x, collections.abc.Sequence)


def print_table(
    table: list[list[Any]], header: Optional[list[Any]] = None, sep: str = "   ", numfmt: str = "{}"
) -> None:
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '{:.2f}'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns.
    """
    justs = ["rjust" if isnumber(x) else "ljust" for x in table[0]]

    if header:
        table.insert(0, header)

    table = [[numfmt.format(x) if isnumber(x) else x for x in row] for row in table]
    sizes = list(
        map(
            lambda seq: max(map(len, seq)),
            list(zip(*[map(str, row) for row in table], strict=False)),
        )
    )

    for row in table:
        print(
            sep.join(
                getattr(str(x), j)(size) for (j, size, x) in zip(justs, sizes, row, strict=False)
            )
        )


def open_data(name: str, mode: str = "r") -> IO[str]:
    aima_root = os.path.dirname(__file__)
    aima_file = os.path.join(aima_root, *["aima-data", name])

    return open(aima_file, mode=mode)


def failure_test(algorithm: Callable[[Any], Any], tests: Iterable[tuple[Any, Any]]) -> float:
    """Grades the given algorithm based on how many tests it passes.
    Most algorithms have arbitrary output on correct execution, which is difficult
    to check for correctness. On the other hand, a lot of algorithms output something
    particular on fail (for example, False, or None).
    tests is a list with each element in the form: (values, failure_output).
    """
    return mean(int(algorithm(x) != y) for x, y in tests)


# ______________________________________________________________________________
# Expressions

# See https://docs.python.org/3/reference/expressions.html#operator-precedence
# See https://docs.python.org/3/reference/datamodel.html#special-method-names


class Expr:
    """A mathematical expression with an operator and 0 or more arguments.
    op is a str like '+' or 'sin'; args are Expressions.
    Expr('x') or Symbol('x') creates a symbol (a nullary Expr).
    Expr('-', x) creates a unary; Expr('+', x, 1) creates a binary.
    """

    def __init__(self, op: Any, *args: Any) -> None:
        self.op: str = str(op)
        self.args: tuple[Any, ...] = args

    # Operator overloads
    def __neg__(self) -> "Expr":
        return Expr("-", self)

    def __pos__(self) -> "Expr":
        return Expr("+", self)

    def __invert__(self) -> "Expr":
        return Expr("~", self)

    def __add__(self, rhs: Any) -> "Expr":
        return Expr("+", self, rhs)

    def __sub__(self, rhs: Any) -> "Expr":
        return Expr("-", self, rhs)

    def __mul__(self, rhs: Any) -> "Expr":
        return Expr("*", self, rhs)

    def __pow__(self, rhs: Any) -> "Expr":
        return Expr("**", self, rhs)

    def __mod__(self, rhs: Any) -> "Expr":
        return Expr("%", self, rhs)

    def __and__(self, rhs: Any) -> "Expr":
        return Expr("&", self, rhs)

    def __xor__(self, rhs: Any) -> "Expr":
        return Expr("^", self, rhs)

    def __rshift__(self, rhs: Any) -> "Expr":
        return Expr(">>", self, rhs)

    def __lshift__(self, rhs: Any) -> "Expr":
        return Expr("<<", self, rhs)

    def __truediv__(self, rhs: Any) -> "Expr":
        return Expr("/", self, rhs)

    def __floordiv__(self, rhs: Any) -> "Expr":
        return Expr("//", self, rhs)

    def __matmul__(self, rhs: Any) -> "Expr":
        return Expr("@", self, rhs)

    def __or__(self, rhs: Any) -> Any:
        """Allow both P | Q, and P |'==>'| Q."""
        if isinstance(rhs, Expression):
            return Expr("|", self, rhs)
        return PartialExpr(rhs, self)

    # Reverse operator overloads
    def __radd__(self, lhs: Any) -> "Expr":
        return Expr("+", lhs, self)

    def __rsub__(self, lhs: Any) -> "Expr":
        return Expr("-", lhs, self)

    def __rmul__(self, lhs: Any) -> "Expr":
        return Expr("*", lhs, self)

    def __rdiv__(self, lhs: Any) -> "Expr":
        return Expr("/", lhs, self)

    def __rpow__(self, lhs: Any) -> "Expr":
        return Expr("**", lhs, self)

    def __rmod__(self, lhs: Any) -> "Expr":
        return Expr("%", lhs, self)

    def __rand__(self, lhs: Any) -> "Expr":
        return Expr("&", lhs, self)

    def __rxor__(self, lhs: Any) -> "Expr":
        return Expr("^", lhs, self)

    def __ror__(self, lhs: Any) -> "Expr":
        return Expr("|", lhs, self)

    def __rrshift__(self, lhs: Any) -> "Expr":
        return Expr(">>", lhs, self)

    def __rlshift__(self, lhs: Any) -> "Expr":
        return Expr("<<", lhs, self)

    def __rtruediv__(self, lhs: Any) -> "Expr":
        return Expr("/", lhs, self)

    def __rfloordiv__(self, lhs: Any) -> "Expr":
        return Expr("//", lhs, self)

    def __rmatmul__(self, lhs: Any) -> "Expr":
        return Expr("@", lhs, self)

    def __call__(self, *args: Any) -> "Expr":
        """Call: if 'f' is a Symbol, then f(0) == Expr('f', 0)."""
        if self.args:
            raise ValueError("Can only do a call for a Symbol, not an Expr")
        return Expr(self.op, *args)

    # Equality and repr
    def __eq__(self, other: Any) -> bool:
        """'x == y' evaluates to True or False; does not build an Expr."""
        return isinstance(other, Expr) and self.op == other.op and self.args == other.args

    def __lt__(self, other: Any) -> bool:
        return isinstance(other, Expr) and str(self) < str(other)

    def __hash__(self) -> int:
        return hash(self.op) ^ hash(self.args)

    def __repr__(self) -> str:
        op = self.op
        args = [str(arg) for arg in self.args]
        if op.isidentifier():  # f(x) or f(x, y)
            return "{}({})".format(op, ", ".join(args)) if args else op
        if len(args) == 1:  # -x or -(x + 1)
            return op + args[0]
        # (x - y)
        opp = " " + op + " "
        return "(" + opp.join(args) + ")"


# An 'Expression' is either an Expr or a Number.
# Symbol is not an explicit type; it is any Expr with 0 args.


Number = (int, float, complex)
Expression = (Expr, Number)


def Symbol(name: str) -> Expr:
    """A Symbol is just an Expr with no args."""
    return Expr(name)


def symbols(names: str) -> tuple[Expr, ...]:
    """Return a tuple of Symbols; names is a comma/whitespace delimited str."""
    return tuple(Symbol(name) for name in names.replace(",", " ").split())


def subexpressions(x: Any) -> Iterable[Any]:
    """Yield the subexpressions of an Expression (including x itself)."""
    yield x
    if isinstance(x, Expr):
        for arg in x.args:
            yield from subexpressions(arg)


def arity(expression: Any) -> int:
    """The number of sub-expressions in this expression."""
    if isinstance(expression, Expr):
        return len(expression.args)
    # expression is a number
    return 0


# For operators that are not defined in Python, we allow new InfixOps:


class PartialExpr:
    """Given 'P |'==>'| Q, first form PartialExpr('==>', P), then combine with Q."""

    def __init__(self, op: str, lhs: Any) -> None:
        self.op: str
        self.lhs: Any
        self.op, self.lhs = op, lhs

    def __or__(self, rhs: Any) -> Expr:
        return Expr(self.op, self.lhs, rhs)

    def __repr__(self) -> str:
        return f"PartialExpr('{self.op}', {self.lhs})"


def expr(x: Any) -> Any:
    """Shortcut to create an Expression. x is a str in which:
    - identifiers are automatically defined as Symbols.
    - ==> is treated as an infix |'==>'|, as are <== and <=>.
    If x is already an Expression, it is returned unchanged. Example:
    >>> expr("P & Q ==> Q")
    ((P & Q) ==> Q)
    """
    if isinstance(x, str):
        return eval(expr_handle_infix_ops(x), defaultkeydict(Symbol))
    return x


infix_ops: list[str] = ["==>", "<==", "<=>"]


def expr_handle_infix_ops(x: str) -> str:
    """Given a str, return a new str with ==> replaced by |'==>'|, etc.
    >>> expr_handle_infix_ops("P ==> Q")
    "P |'==>'| Q"
    """
    for op in infix_ops:
        x = x.replace(op, "|" + repr(op) + "|")
    return x


class defaultkeydict(collections.defaultdict[Any, Any]):
    """Like defaultdict, but the default_factory is a function of the key.
    >>> d = defaultkeydict(len)
    ... d["four"]
    4
    """

    def __init__(self, default_factory: Callable[[Any], Any]) -> None:
        super().__init__()
        self.key_default_factory: Callable[[Any], Any] = default_factory

    def __missing__(self, key: Any) -> Any:
        self[key] = result = self.key_default_factory(key)
        return result


class hashabledict(dict[Any, Any]):
    """Allows hashing by representing a dictionary as tuple of key:value pairs.
    May cause problems as the hash value may change during runtime.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return 1


# ______________________________________________________________________________
# Monte Carlo tree node and ucb function


class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, parent: Any = None, state: Any = None, U: float = 0.0, N: int = 0) -> None:
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children: Dict[Any, Any] = {}
        self.actions: Any = None


def ucb(n: Any, C: float = 1.4) -> float:
    return float(np.inf if n.N == 0 else n.U / n.N + C * np.sqrt(np.log(n.parent.N) / n.N))


# ______________________________________________________________________________
# Useful Shorthands


class Bool(int):
    """Just like `bool`, except values display as 'T' and 'F' instead of 'True' and 'False'."""

    __str__ = __repr__ = lambda self: "T" if self else "F"


T = Bool(True)
F = Bool(False)
