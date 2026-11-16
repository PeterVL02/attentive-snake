"""Input stages: the part of a policy that reads the observation, chosen by observation kind.

A policy describes *what* its input stage must produce (tokens, one feature vector, one
vector per frame); ``catalog`` picks the modules that do it for the observation kind at
hand (``grid``, ``vector`` or ``image``). Policies never branch on the observation kind themselves.
"""
