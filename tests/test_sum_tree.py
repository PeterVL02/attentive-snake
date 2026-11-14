"""The sum tree: totals, prefix-sum lookup against a brute-force cumulative sum, boundaries, repeated updates."""

import numpy as np
import pytest

from attentive_snake.training.dqn.sum_tree import SumTree


@pytest.mark.parametrize("capacity", [1, 2, 5, 8, 13, 100])  # powers of two and not
def test_total_and_lookup_agree_with_a_cumulative_sum(capacity: int) -> None:
    rng = np.random.default_rng(capacity)
    values = rng.random(capacity) * 3
    values[rng.random(capacity) < 0.2] = 0.0  # zero leaves are never found
    values[0] = 1.0
    tree = SumTree(capacity)
    tree.set(np.arange(capacity), values)
    assert tree.total == pytest.approx(values.sum())
    assert np.array_equal(tree[np.arange(capacity)], values)
    prefixes = rng.random(2000) * tree.total
    expected = np.searchsorted(np.cumsum(values), prefixes, side="right")
    found = tree.find(prefixes)
    assert np.array_equal(found, expected)
    assert (values[found] > 0).all()


def test_boundaries_of_the_prefix_range() -> None:
    tree = SumTree(5)
    tree.set(np.arange(5), np.array([0.0, 2.0, 0.0, 3.0, 0.0]))
    assert tree.find(np.array([0.0, 1.999, 2.0, 4.999])).tolist() == [1, 1, 3, 3]
    assert tree.find(np.array([np.nextafter(tree.total, 0.0)])).tolist() == [3]
    assert tree.find(np.array([tree.total, tree.total + 1.0])).tolist() == [3, 3]  # rounding past the end stays on a real leaf
    assert SumTree(1).find(np.array([0.0])).tolist() == [0]


def test_updates_change_the_total_and_a_repeated_leaf_keeps_its_last_value() -> None:
    tree = SumTree(6)
    tree.set(np.arange(6), np.ones(6))
    tree.set(np.array([2, 4, 2]), np.array([5.0, 0.0, 7.0]))
    assert tree[np.array([2, 4])].tolist() == [7.0, 0.0]
    assert tree.total == 1 + 1 + 7 + 1 + 0 + 1
    assert tree.find(np.array([2.5])).tolist() == [2] and tree.find(np.array([9.5])).tolist() == [3]


def test_many_updates_keep_the_total_exact() -> None:
    rng = np.random.default_rng(0)
    tree = SumTree(37)
    leaves = np.zeros(37)
    for _ in range(200):
        index = rng.integers(0, 37, size=8)
        value = rng.random(8)
        tree.set(index, value)
        leaves[index] = value  # numpy: last duplicate wins, like the tree
    assert tree.total == pytest.approx(leaves.sum(), rel=1e-12)
