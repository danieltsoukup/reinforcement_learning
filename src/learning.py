from gym.core import Env
from src.policies import (
    StateActionListRecord,
    StateActionRecord,
    TabularGreedyPolicy,
    TabularEpsilonGreedyPolicy,
)
from collections import defaultdict
import numpy as np
from typing import List


class MonteCarlo(object):
    """Monte Carlo control with exploring starts for tabular state-action value learning.

    Ref: Sutton & Barto Chapter 5.3
    """

    def __init__(self, environment: Env) -> None:
        self.environment = environment
        self.policy = TabularEpsilonGreedyPolicy(self.environment, 0.5)
        self.returns = StateActionListRecord()

    def learn(self, num_episodes: int) -> np.ndarray:
        """Repeatedly runs episodes and updates the state-action values returning the list of total rewards."""
        total_rewards = []
        for _ in range(num_episodes):
            episode_rewards = self._run_episode()
            self._update_state_action_values()
            total_rewards.append(episode_rewards.sum())

        return np.array(total_rewards)

    def _get_first_state_action(self):
        state = self.environment.observation_space.sample()
        action = self.environment.action_space.sample()

        return np.array(state), action

    def _run_episode(self, first_state=None, first_action=None) -> None:
        """Run a single episode and record rewards after the first occurrance of each state-action pair.

        Optionally, the user can provide the first state-action pair.
        """

        state_action_first_occurrence_idx = StateActionRecord(int)
        episode_rewards = []

        state = self.environment.reset()
        done = False

        if first_state is not None:
            state = first_state

        first_step = True
        step_count = 0

        while done is False:
            if first_step is True and first_action is not None:
                action = first_action
                first_step = False
            else:
                action = self.policy.select_action(state)

            if state_action_first_occurrence_idx.contains(state, action) is False:
                state_action_first_occurrence_idx.set(state, action, step_count)

            state, reward, done, info = self.environment.step(action)

            episode_rewards.append(reward)

            step_count += 1

        for state, action in state_action_first_occurrence_idx.raw_state_actions:
            first_occurrance_idx = state_action_first_occurrence_idx.get(state, action)
            state_action_return = sum(episode_rewards[first_occurrance_idx:])
            self.returns.set(state, action, state_action_return)

        return np.array(episode_rewards)

    def _update_state_action_values(self) -> None:
        for state, action in self.returns.raw_state_actions:
            reward_list = self.returns.get(state, action)
            mean_reward = self._get_mean_reward(reward_list)

            self.policy.set_state_action_value(state, action, mean_reward)

    def _get_mean_reward(self, reward_list: List[float]) -> float:
        if len(reward_list) > 0:
            mean_reward = np.mean(reward_list)
        else:
            mean_reward = 0

        return mean_reward
