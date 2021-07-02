from gym.core import Env
from gym.spaces import Space
from abc import abstractmethod
import numpy as np
from collections import defaultdict


class Policy(object):
    """Base class for control policies."""

    @abstractmethod
    def select_action(self, state: Space, environment: Env):
        pass


class ConstantPolicy(Policy):
    """Always play the same action."""

    def __init__(self, action) -> None:
        self.action = action

    def select_action(self, state: Space, environment: Env):
        return self.action


class RandomPolicy(Policy):
    def select_action(self, state: Space, environment: Env):
        return environment.action_space.sample()


class TabularEpsilonGreedyPolicy(Policy):
    """Base class for table based control policies."""

    def __init__(self, enviroment: Env, eps: float) -> None:
        self.environment = enviroment
        self.state_action_values = defaultdict(float)
        self.eps = eps

    def select_action(self, state: Space):
        """With prob eps, returns a random action (explore), or 1-eps prob the greedy action
        using the state-action values.
        """

        explore = np.random.choice([True, False], p=[self.eps, 1 - self.eps])

        if explore:
            action = self.environment.action_space.sample()
        else:
            self._greedy_action(state, self.environment)

        return action

    def _greedy_action(self, state: Space):
        """Returns the action with the highest value for the state."""
        best_action = max(
            self.environment.action_space,
            key=lambda action: self.state_action_values((state, action)),
        )

        return best_action
