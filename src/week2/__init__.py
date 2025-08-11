class PriorityQueue:
    def __init__(self):
        self._heap = []

    def push(self, item, priority):
        """Add an item to the priority queue with the given priority.

        Parameters
        ----------
            item: The item to add to the queue.
            priority: The priority of the item. Lower numbers indicate higher priority.

        """
        heapq.heappush(self._heap, (priority, item))

    def pop(self):
        """Remove and return the item with the highest priority (lowest priority number).

        Returns:
            The item with the highest priority.

        Raises:
            IndexError: If the priority queue is empty.

        """
        if self.is_empty():
            raise IndexError("pop from an empty priority queue")
        return heapq.heappop(self._heap)[1]

    def peek(self):
        """Return the item with the highest priority without removing it.

        Returns:
            The item with the highest priority.

        Raises:
            IndexError: If the priority queue is empty.

        """
        if self.is_empty():
            raise IndexError("peek from an empty priority queue")
        return self._heap[0][1]

    def is_empty(self):
        """Check if the priority queue is empty.

        Returns:
            True if the priority queue is empty, False otherwise.

        """
        return len(self._heap) == 0

    def __len__(self):
        """Return the number of items in the priority queue.

        Returns:
            The number of items in the queue.

        """
        return len(self._heap)
