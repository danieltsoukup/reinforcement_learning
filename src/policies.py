from gym.core import Env
from gym.spaces import Space
from abc import abstractmethod
import numpy as np
from collections import defaultdict


class Policy(object):
    """Base class for control policies."""

    def __init__(self, environment: Env) -> None:
        self.environment = environment

    @abstractmethod
    def select_action(self, state: Space):
        pass


class ConstantPolicy(Policy):
    """Always play the same action."""

    def __init__(self, environment: Env, action) -> None:
        super().__init__(environment)
        self.action = action

    def select_action(self, state: Space):
        return self.action


class RandomPolicy(Policy):
    def __init__(self, environment: Env) -> None:
        super().__init__(environment)

    def select_action(self, state: Space):
        return self.environment.action_space.sample()


class TabularGreedyPolicy(Policy):
    """Base class for table based control policies."""

    def __init__(self, environment: Env) -> None:
        super().__init__(environment)
        self.state_action_values = defaultdict(float)

    def set_state_action_value(self, state, action, new_value: float) -> None:
        state_action_key = (str(state), action)
        self.state_action_values[state_action_key] = new_value

    def select_action(self, state: np.ndarray):
        """With prob eps, returns a random action (explore), or 1-eps prob the greedy action
        using the state-action values.
        """

        return self._greedy_action(state)

    def _greedy_action(self, state: np.ndarray):
        """
        Returns the action with the highest value for the state.

        If no action has a recorded value for the state, we select a random action.
        """

        all_state_action_values = self.state_action_values.items()
        current_state_action_values = [
            item for item in all_state_action_values if item[0][0] == str(state)
        ]

        if len(current_state_action_values) > 0:
            best_state_action_value = max(
                current_state_action_values,
                key=lambda item: item[1],
            )

            best_action = best_state_action_value[0][1]
        else:
            best_action = self.environment.action_space.sample()

        return best_action


class TabularEpsilonGreedyPolicy(TabularGreedyPolicy):
    """Epsilon-greedy policy based on state-action values."""

    def __init__(self, environment: Env, eps: float) -> None:
        super().__init__(environment)
        self.eps = eps

    def select_action(self, state: np.ndarray):
        """With prob eps, returns a random action (explore), or 1-eps prob the greedy action
        using the state-action values.
        """

        explore = np.random.choice([True, False], p=[self.eps, 1 - self.eps])

        if explore:
            action = self.environment.action_space.sample()
        else:
            action = self._greedy_action(state, self.environment)

        return action
