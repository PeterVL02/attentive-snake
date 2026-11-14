"""A sum tree over a fixed number of leaves: O(log n) priority updates and prefix-sum lookups, vectorised over a batch.

Pure numpy; it knows nothing about transitions. Leaves live at the bottom of a binary
heap laid out in one array (node ``n`` has children ``2n`` and ``2n + 1``, the root is node 1);
the leaf count is rounded up to a power of two and the extra leaves stay 0, so any capacity works.
"""

import numpy as np


class SumTree:
    """``capacity`` non-negative leaf values (all 0 at first) and the sums of every subtree of them."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive.")
        self._capacity = capacity
        self._first_leaf = 1 << (capacity - 1).bit_length()  # smallest power of two >= capacity
        self._nodes = np.zeros(2 * self._first_leaf, dtype=np.float64)

    @property
    def total(self) -> float:
        """The sum of all leaves."""
        return float(self._nodes[1])

    def __getitem__(self, leaves: np.ndarray) -> np.ndarray:
        """The values of ``leaves`` (an index array)."""
        return self._nodes[self._first_leaf + leaves]

    def set(self, leaves: np.ndarray, values: np.ndarray) -> None:
        """Assign ``values`` to ``leaves`` (index arrays of equal length; for a repeated leaf the last value counts)."""
        nodes = self._first_leaf + leaves
        self._nodes[nodes] = values
        while nodes[0] > 1:  # every touched node sits on the same level; each parent is recomputed from its children (no drift)
            nodes = np.unique(nodes >> 1)
            self._nodes[nodes] = self._nodes[2 * nodes] + self._nodes[2 * nodes + 1]

    def find(self, prefix_sums: np.ndarray) -> np.ndarray:
        """For each prefix sum ``s`` in ``[0, total)`` the leaf ``i`` with ``sum(leaves[:i]) <= s < sum(leaves[:i + 1])``.

        Never returns a leaf whose value is 0, even if rounding pushes ``s`` up to or past the total.
        """
        remaining = np.array(prefix_sums, dtype=np.float64)
        nodes = np.ones(len(remaining), dtype=np.int64)
        while nodes[0] < self._first_leaf:  # every sample sits on the same level
            left = 2 * nodes
            left_sum = self._nodes[left]
            go_right = (remaining >= left_sum) & (self._nodes[left + 1] > 0.0)
            remaining -= np.where(go_right, left_sum, 0.0)
            nodes = left + go_right
        return nodes - self._first_leaf
