from gym.core import Env
from src.policies import TabularGreedyPolicy
from collections import defaultdict
import numpy as np
from typing import List


class FirstTimeMarker(defaultdict):
    """Utility object to mark if a key was used before."""

    def __init__(self):
        super().__init__(lambda: True)

    def pop(self, key) -> bool:
        is_first = self[key]
        self[key] = False

        return is_first


class MonteCarlo(object):
    """Monte Carlo control for tabular state-action value learning.

    Ref: Sutton & Barto Chapter 5.3
    """

    def __init__(self, environment: Env) -> None:
        self.environment = environment
        self.policy = TabularGreedyPolicy(self.environment)
        self.returns = defaultdict(list)

    def learn(self, num_episodes: int) -> None:
        for _ in range(num_episodes):
            self._run_episode()
            self._update_state_action_values()

    def _run_episode(self) -> None:
        """Run a single episode and record rewards after the first occurrance of each state-action pair."""

        first_time_marker = FirstTimeMarker()

        state = self.environment.reset()
        done = False

        while done is False:
            action = self.policy.select_action(state)

            state_action_tuple = (str(state), action)
            is_first_time = first_time_marker.pop(state_action_tuple)

            state, reward, done, info = self.environment.step(action)

            if is_first_time is True:
                self.returns[state_action_tuple].append(reward)

    def _update_state_action_values(self) -> None:
        for state_action, reward_list in self.returns.items():
            mean_reward = self._get_mean_reward(reward_list)
            state, action = state_action
            self.policy.set_state_action_value(state, action, mean_reward)

    def _get_mean_reward(self, reward_list: List[float]) -> float:
        if len(reward_list) > 0:
            mean_reward = np.mean(reward_list)
        else:
            mean_reward = 0

        return mean_reward
