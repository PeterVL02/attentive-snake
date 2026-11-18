"""The transitions an auxiliary objective learns from, as the policy's encoder reads them.

An objective feeds its encoders directly, past ``BasePolicy.forward``, so the rollout's raw
observations must be normalised for it.
"""

import dataclasses

from attentive_snake.models.auxiliary import TransitionBatch
from attentive_snake.models.observation_normalizer import ObservationNormalizer


def normalized_transitions(normalizer: ObservationNormalizer, batch: TransitionBatch) -> TransitionBatch:
    """``batch`` with every observation in it normalised."""
    sequences = batch.sequences
    if sequences is not None:
        sequences = dataclasses.replace(sequences, next_observations=normalizer(sequences.next_observations))
    return dataclasses.replace(
        batch,
        observations=normalizer(batch.observations),
        next_observations=normalizer(batch.next_observations),
        sequences=sequences,
    )
