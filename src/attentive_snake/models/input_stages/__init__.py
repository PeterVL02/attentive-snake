"""Input stages: the part of a policy that reads the observation, chosen by observation kind.

A policy describes *what* its input stage must produce (tokens, one feature vector, one
vector per frame); ``catalog`` picks the modules that do it for the observation kind at
hand (``grid`` or ``vector``). Policies never branch on the observation kind themselves.
An image stem (e.g. patches as tokens) would be one more module ``image.py`` and one more
entry per table in ``catalog``.
"""
