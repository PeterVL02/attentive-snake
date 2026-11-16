"""Reward search: adjusting the reward table a policy trains on, judged only by the game's real outcomes.

A population search in the style of population-based training of reward weights: starting from a
parent run's checkpoint and its ``[reward]`` table, each *generation* forks the current best
checkpoint ``population`` times and trains every fork ``steps_per_member`` env steps. Member 0
keeps the current reward table, the others train on a perturbed copy (only the selected fields
change, within bounds: ``perturbation``). When all have ended, each member's final greedy
evaluation is read, and the member with the best value of the objective statistic (one of the
game's own statistics, never the training return) wins; its checkpoint and its table start the
next generation. Ties go to the unchanged member.

The search only forks runs and reads their evaluations, so it works for every game with numeric
reward coefficients and for every algorithm. It talks to training through ``backend.TrainingBackend``
(the app implements it with the training queue); ``searches.RewardSearches`` is the entry point.

Modules, one job each: ``settings`` (the tunable numbers), ``spec`` (what one search is),
``reward_fields`` (which coefficients can be searched), ``objective`` (which statistic judges),
``perturbation`` (seeded new tables), ``selection`` (the winner), ``naming`` (member run names),
``member_config`` (a member's experiment tables), ``state`` (what is remembered, as JSON),
``storage`` (the files), ``generation`` (planning one), ``member_progress`` (one member's next
state), ``driver`` (one step of a search), ``searches`` (all searches, thread-safe).
"""
