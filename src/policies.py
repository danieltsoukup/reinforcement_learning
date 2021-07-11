from gym.core import Env
from gym.spaces import Space
from abc import abstractmethod
import numpy as np
from collections import defaultdict
from typing import Any


class StateActionRecord(object):
    def __init__(self, type_):
        self.type_ = type_
        self.record = defaultdict(lambda: defaultdict(self.type_))
        self.state_string_reverse = dict()
        self.action_string_reverse = dict()
        self.raw_state_actions = []

    def contains(self, state: Any, action: Any) -> bool:
        state_string = self._state_to_string(state)
        action_string = self._action_to_string(action)

        check_state = state_string in self.record.keys()
        check_action = action_string in self.record[state_string].keys()

        return check_action and check_state

    def get(self, state: Any, action: Any) -> Any:
        state_string = self._state_to_string(state)
        action_string = self._action_to_string(action)

        return self.record[state_string][action_string]

    def set(self, state: Any, action: Any, value: Any) -> None:
        state_string = self._state_to_string(state)
        action_string = self._action_to_string(action)

        if self.contains(state, action) is False:
            self.raw_state_actions.append((state, action))

        self.record[state_string][action_string] = value

    def _state_to_string(self, state: Any) -> str:
        state_string = str(state)
        if state_string not in self.state_string_reverse.keys():
            self.state_string_reverse[state_string] = state

        return state_string

    def _action_to_string(self, action: Any) -> str:
        action_string = str(action)
        if action_string not in self.action_string_reverse.keys():
            self.action_string_reverse[action_string] = action

        return action_string

    def get_state_record(self, state: Any) -> defaultdict:
        state_string = self._state_to_string(state)

        return self.record[state_string]

    def greedy_action(self, state: Any, default_action: Any) -> Any:
        """
        Returns the action with the highest value for the state
        (if no values were entered, the default is returned).

        If no action has a recorded value for the state, we select a random action.
        """

        state_action_dict = self.get_state_record(state)

        if len(state_action_dict) > 0:
            best_action_key = max(
                state_action_dict.keys(),
                key=lambda key_: state_action_dict[key_],
            )
            best_action = self.action_string_reverse[best_action_key]

        else:
            best_action = default_action

        return best_action


class StateActionListRecord(StateActionRecord):
    def __init__(self):
        super().__init__(list)

    def set(self, state: Any, action: Any, value):
        """The values are appended to the lists stored for each state-value pair."""
        state_string = self._state_to_string(state)
        action_string = self._action_to_string(action)

        if self.contains(state, action) is False:
            self.raw_state_actions.append((state, action))

        self.record[state_string][action_string].append(value)


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
        self.state_action_values = StateActionRecord(float)

    def set_state_action_value(self, state, action, new_value: float) -> None:
        self.state_action_values.set(state, action, new_value)

    def select_action(self, state: np.ndarray):
        """With prob eps, returns a random action (explore), or 1-eps prob the greedy action
        using the state-action values.
        """
        default_action = self.environment.action_space.sample()

        return self.state_action_values.greedy_action(state, default_action)


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
            action = super().select_action(state)

        return action
