"""Making the package's game of a title (``minatar.Environment``).

``minatar`` is imported here, on first use: it imports matplotlib and seaborn (for its own
display, which is never used), so importing the game package must not import it.
"""

from attentive_snake.games.minatar.config import MinAtarRules


def make_emulator(rules: MinAtarRules):
    """``minatar.Environment`` of ``rules.title`` with the rules' sticky actions and difficulty ramping.

    Its ``env`` is the title's own game (``channels``, ``minimal_action_set()``), ``act(a)`` applies
    sticky actions and returns ``(points, terminal)``, ``seed(s)`` reseeds game and sticky actions.
    """
    from minatar import Environment

    return Environment(rules.title, sticky_action_prob=rules.sticky_action_prob, difficulty_ramping=rules.difficulty_ramping)
