"""Leaving the global random streams (Python, numpy, torch) exactly as they were around a block of code."""

import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def preserved_rng_state() -> Iterator[None]:
    """Whatever the block draws from the global generators, they continue afterwards as if it had not run.

    Used around evaluations during training, so evaluating does not change the training's
    action sampling. Not thread-safe: other threads using the global generators meanwhile
    are affected (do not use it in a server).
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng():
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
