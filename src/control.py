from gym.core import Env
from src.policies import (
    StateActionListRecord,
    StateActionRecord,
    TabularGreedyPolicy,
)
import numpy as np
from typing import Any, List
from abc import abstractmethod


class TabularControlMethod(object):
    """Base class for tabular control methods."""

    def __init__(self, environment: Env, tabular_policy: TabularGreedyPolicy) -> None:
        self.environment = environment
        self.policy = tabular_policy

    @abstractmethod
    def learn(self, num_episodes: int) -> np.ndarray:
        """Repeatedly runs episodes and updates the state-action values returning the list of total rewards."""
        pass

    @abstractmethod
    def _update_state_action_values(self) -> None:
        pass

    @abstractmethod
    def _run_episode(self) -> np.ndarray:
        pass


class MonteCarlo(TabularControlMethod):
    """
    Monte Carlo control with exploring starts for tabular state-action value learning.

    Ref: Sutton & Barto Chapter 5.3
    """

    def __init__(self, environment: Env, tabular_policy: TabularGreedyPolicy) -> None:
        super().__init__(environment, tabular_policy)
        self.returns = StateActionListRecord()

    def __str__(self) -> str:
        return "MonteCarlo"

    def _get_first_state_action(self):
        state = self.environment.observation_space.sample()
        action = self.environment.action_space.sample()

        return np.array(state), action

    def learn(self, num_episodes: int) -> np.ndarray:
        """Repeatedly runs episodes and updates the state-action values returning the list of total rewards."""
        total_rewards = []
        for _ in range(num_episodes):
            episode_rewards = self._run_episode()
            total_rewards.append(episode_rewards.sum())

            self._update_state_action_values()

        return np.array(total_rewards)

    def _run_episode(self, first_state=None, first_action=None) -> np.ndarray:
        """Run a single episode and record rewards after the first occurrance of each state-action pair.

        Optionally, the user can provide the first state-action pair.
        """

        state_action_first_occurrence_idx = StateActionRecord(int)
        episode_rewards = []

        state = self.environment.reset()
        done = False

        if first_state is not None:
            state = first_state

        step_count = 0

        while done is False:
            if step_count == 0 and first_action is not None:
                action = first_action
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

    @staticmethod
    def _get_mean_reward(reward_list: List[float]) -> float:
        if len(reward_list) > 0:
            mean_reward = np.mean(reward_list)
        else:
            mean_reward = 0

        return mean_reward


class SARSA(TabularControlMethod):
    """
    SARSA or TD(0) for control.
    """

    def __init__(
        self, environment: Env, policy: TabularGreedyPolicy, alpha: float, gamma: float
    ):
        super().__init__(environment, policy)
        self.alpha = alpha
        self.gamma = gamma

    def learn(self, num_episodes: int) -> np.ndarray:
        """
        Repeatedly runs episodes and return the list of total rewards per episode.
        """
        total_rewards = []
        for _ in range(num_episodes):
            episode_rewards = self._run_episode()
            total_rewards.append(episode_rewards.sum())

        return np.array(total_rewards)

    def _run_episode(self) -> np.ndarray:
        state = self.environment.reset()

        done = False

        episode_rewards = []
        while not done:
            action = self.policy.select_action(state)
            next_state, reward, done, info = self.environment.step(action)
            next_action = self.policy.select_action(next_state)

            self._update_state_action_values(
                state, action, reward, next_state, next_action
            )

            episode_rewards.append(reward)
            state = next_state

        return np.array(episode_rewards)

    def _update_state_action_values(
        self, state: Any, action: Any, reward: Any, next_state: Any, next_action: Any
    ) -> None:
        current_state_action_value = self.policy.state_action_values.get(state, action)
        next_state_action_value = self.policy.state_action_values.get(
            next_state, next_action
        )

        updated_state_action_value = current_state_action_value + self.alpha * (
            reward + self.gamma * next_state_action_value - current_state_action_value
        )

        self.policy.set_state_action_value(state, action, updated_state_action_value)
