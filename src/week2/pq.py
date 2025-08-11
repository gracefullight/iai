"""Priority queue utility for search algorithms."""

from __future__ import annotations

import heapq


class PriorityQueue[T]:
    """A min-heap based priority queue.

    Lower numeric priority values are treated as higher priority.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[float, T]] = []

    def push(self, item: T, priority: float) -> None:
        """Add an item with the given priority."""
        heapq.heappush(self._heap, (priority, item))

    def pop(self) -> T:
        """Remove and return the item with the highest priority.

        Raises IndexError if the queue is empty.
        """
        if self.is_empty():
            msg = "pop from an empty priority queue"
            raise IndexError(msg)
        return heapq.heappop(self._heap)[1]

    def peek(self) -> T:
        """Return the item with the highest priority without removing it.

        Raises IndexError if the queue is empty.
        """
        if self.is_empty():
            msg = "peek from an empty priority queue"
            raise IndexError(msg)
        return self._heap[0][1]

    def is_empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self._heap) == 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        """Return the number of items in the queue."""
        return len(self._heap)
