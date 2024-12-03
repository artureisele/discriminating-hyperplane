from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

import numpy as np

import gymnasium
from gymnasium import spaces, Wrapper, Env
from gymnasium.utils import RecordConstructorArgs, seeding

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec, WrapperSpec

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

class RewardWrapperHalfcheetahHyperPlane(Wrapper[ObsType, ActType, ObsType, ActType]):


    def __init__(self, env: Env[ObsType, ActType], safety_reward = False):
        """Constructor for the Reward wrapper.

        Args:
            env: Environment to be wrapped.
        """
        Wrapper.__init__(self, env)
        self.safety_reward = safety_reward

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        bonus =  int(self.action_space.contains(action))
        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation[0] >= -0.3:
            if self.safety_reward:
                reward = 1
            terminated = False
        else:
            if self.safety_reward:
                reward = -100
            terminated = True
        observation = observation.astype(np.float32)
        return observation, reward, terminated, truncated, info