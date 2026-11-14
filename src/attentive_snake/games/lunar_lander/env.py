"""Lunar Lander as a ``GameEnv``: Gymnasium's ``LunarLander-v3`` with discrete actions, 8 numbers per frame.

Actions: 0 nothing, 1 left engine, 2 main engine, 3 right engine. Observation (see
``game.OBSERVATION_LABELS``): position, velocity, angle, angular velocity, leg contacts.
Rendering is lazy: the Gymnasium env is made without a render mode, and only the first
``view()`` switches it to ``"rgb_array"`` (engine particles, Gymnasium's only rendering
cost in ``step``, are decoration without effect on the physics). Training never calls
``view()``, so it never renders. With ``LanderRules.hide_velocities`` the policy sees zeros for the
three velocity numbers (``hidden_velocities``); reward, score and outcome use the true state.
"""

from typing import Any

import numpy as np

from attentive_snake.env.types import EnvStep
from attentive_snake.games.lunar_lander.config import LanderRules
from attentive_snake.games.lunar_lander.hidden_velocities import visible_part
from attentive_snake.games.lunar_lander.outcome import step_event
from attentive_snake.games.lunar_lander.reward import LanderReward, lander_reward
from attentive_snake.views.image import image_payload

GYMNASIUM_ID = "LunarLander-v3"


def make_gymnasium_env(rules: LanderRules):
    """The Gymnasium env (with its ``TimeLimit``), no render mode."""
    import gymnasium  # here, so that finding the game does not import Gymnasium and Box2D

    return gymnasium.make(
        GYMNASIUM_ID,
        max_episode_steps=rules.max_episode_steps,
        disable_env_checker=True,
        continuous=False,
        gravity=rules.gravity,
        enable_wind=rules.enable_wind,
        wind_power=rules.wind_power,
        turbulence_power=rules.turbulence_power,
    )


class LunarLanderEnv:
    """One Lunar Lander; ``score`` is the episode's raw Gymnasium return, ``reward`` the scaled one."""

    observation_shape = (8,)
    num_actions = 4

    def __init__(self, rules: LanderRules, reward: LanderReward, seed: int | None = None) -> None:
        self._env = make_gymnasium_env(rules)
        self._reward = reward
        self._visible = visible_part(rules)
        self._seed = seed  # used by the first reset that is given no seed
        self._return = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        seed, self._seed = (seed if seed is not None else self._seed), None
        observation, _ = self._env.reset(seed=seed)
        self._return = 0.0
        return self._visible(np.asarray(observation, dtype=np.float32))

    def step(self, action: int) -> EnvStep:
        observation, reward, terminated, truncated, _ = self._env.step(int(action))
        self._return += float(reward)
        timed_out = bool(truncated and not terminated)
        return EnvStep(
            observation=self._visible(np.asarray(observation, dtype=np.float32)),
            reward=lander_reward(self._reward, float(reward), timed_out),
            terminated=bool(terminated),
            truncated=timed_out,
            event=step_event(self._env.unwrapped, float(observation[0]), bool(terminated), bool(truncated)),
            score=self._return,
        )

    def view(self) -> dict[str, Any]:
        lander = self._env.unwrapped
        if lander.render_mode is None:
            lander.render_mode = "rgb_array"
        return image_payload(self._env.render())
