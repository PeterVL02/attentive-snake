"""Evaluating a policy: playing full games greedily and turning them into statistics.

Modules (import them directly; this package imports nothing, so the torch-free parts
stay cheap to import for game packages and config code):

* ``config``: ``EvalConfig``, the ``[evaluation]`` section of an experiment.
* ``settings``: ``EvalSettings``, everything that determines an evaluation's numbers.
* ``generic_statistics``: the statistics every game gets; ``statistics``: generic + the game's own.
* ``player``: plays N episodes with greedy actions (``models.acting``) on fresh environments (torch).
* ``checkpoint_evaluation``: evaluate one checkpoint of a run directory (CLI and app).
* ``record`` / ``log``: one evaluation as JSON, and the run's ``evaluations.jsonl``.
* ``matching``: does a new evaluation reproduce a recorded one.
* ``schedule`` / ``hook``: when and how training evaluates (a ``Trainer`` hook).
* ``rng``: keeps the global random streams untouched around an evaluation.
* ``table``: a plain-text table of an evaluation for the terminal.
"""
