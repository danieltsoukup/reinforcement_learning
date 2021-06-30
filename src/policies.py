from gym.core import Env
from gym.spaces import Space
from abc import abstractmethod


class Policy(object):
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
