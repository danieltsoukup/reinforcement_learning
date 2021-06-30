from gym.core import Env
from gym.spaces import Space
from abc import abstractmethod
import numpy as np


class Policy(object):
    @abstractmethod
    def select_action(self, state: Space, environment: Env):
        pass


class RandomPolicy(Policy):
    def select_action(self, state: Space, environment: Env):
        return environment.action_space.sample()


class Experiment(object):
    def __init__(self, environment: Env, policy: Policy) -> None:
        self.environment = environment
        self.policy = policy

    def run_episode(self):
        state = self.environment.reset()
        done = False

        rewards = []
        while done is False:
            action = self.policy.select_action(state, self.environment)
            state, reward, done, info = self.environment.step(action)
            rewards.append(reward)

        return np.array(rewards)

    def evaluate(self, num_episodes: int) -> float:
        summed_rewards = []
        for _ in range(num_episodes):
            rewards = self.run_episode()
            summed_rewards.append(rewards.sum())

        return summed_rewards.mean()
