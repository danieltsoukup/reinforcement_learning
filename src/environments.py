from gym.core import Env
from gym.spaces import Discrete, Space, Tuple as TupleSpace, Box
from typing import Tuple, List, Dict, Any, Set
import numpy as np
from enum import Enum


class QueueAccessControl(Env):
    REJECT_ACTION = 0
    ACCEPT_ACTION = 1

    def __init__(
        self,
        num_servers: int,
        customer_rewards: List[int],
        customer_probs: List[float],
        queue_size: int,
        unlock_proba: float,
    ) -> None:
        """
        Queue Access Control Environment

            0: accept request
            1: reject request

        Observation space:
            number of available servers and next customer in line


        Args:
            num_servers (int): total number of servers
            customer_rewards (List[int]): rewards for each request type
        """
        super().__init__()

        self.num_servers = num_servers
        self.customer_rewards = customer_rewards
        self.num_customer_types = len(self.customer_rewards)
        self.customer_probs = customer_probs
        self.queue_size = queue_size
        self.unlock_proba = unlock_proba

        self.action_space: Space = Discrete(2)
        self.observation_space: Space = TupleSpace(
            (Discrete(self.num_servers + 1), Discrete(self.num_customer_types))
        )
        self.reward_range: Tuple[int, int] = (
            min(self.customer_rewards),
            max(customer_rewards),
        )

        self.num_free_servers = None
        self.queue = None
        self.current_customer_pointer = None

    def __str__(self) -> str:
        return "QueueAccessControl"

    @property
    def current_customer(self) -> int:
        if self.current_customer_pointer < len(self.queue):
            current_customer = self.queue[self.current_customer_pointer]
        else:
            current_customer = None

        return current_customer

    @property
    def state(self) -> np.ndarray:
        return np.array([self.num_free_servers, self.current_customer])

    def move_customer_pointer(self) -> int:
        """Move the queue pointer and return the next customer.

        Returns:
            int: element of the queue at the new pointer position
        """
        assert (
            len(self.queue) > self.current_customer_pointer
        ), "Cannot move pointer, at the end of the queue already."

        self.current_customer_pointer += 1

    def step(
        self, action: int, unlock: bool = True
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Takes the given action, updates and returns the state with
        the reward, indicator if the episode is done and an info dictionary.
        """
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid."

        info = self._get_step_info()

        if action == type(self).REJECT_ACTION or self.num_free_servers == 0:
            reward = 0

        elif action == type(self).ACCEPT_ACTION:
            reward = self.customer_rewards[self.current_customer]
            self.lock_servers(1)

        self.move_customer_pointer()

        if unlock:
            self.unlock_servers(self.unlock_proba)

        done = self.is_done()

        return self.state, reward, done, info

    def _get_step_info(self):
        info = {
            "current_customer": self.current_customer,
            "current_customer_pointer": self.current_customer_pointer,
        }

        return info

    def is_done(self) -> bool:
        """
        Check if the episode ended by moving past the last queue element.
        """

        return self.current_customer_pointer == self.queue_size

    def lock_servers(self, num_servers_to_lock: int) -> None:
        assert (
            self.num_free_servers >= num_servers_to_lock
        ), f"There is not enough servers to lock {num_servers_to_lock}"

        self.num_free_servers -= num_servers_to_lock

    def unlock_servers(self, unlock_probability) -> None:
        """Unlock each occupied server with the given probability."""

        num_locked_servers = self.num_servers - self.num_free_servers
        num_unlock = np.random.choice(
            2, size=num_locked_servers, p=[1 - unlock_probability, unlock_probability]
        )

        self.num_free_servers += num_unlock.sum()

    def reset(self) -> np.ndarray:
        """Initialize the queue, unlock all servers and setup the state.

        Returns:
            np.ndarray: state after reset
        """
        self.queue = np.random.choice(
            self.num_customer_types, p=self.customer_probs, size=self.queue_size
        )

        self.num_free_servers = self.num_servers

        self.current_customer_pointer = 0

        return self.state


class LineWorld(Env):
    """
    Step left or right on a line graph until reaching either endpoints.

    Each step to non-terminal nodes gives a reward of -1 as well as the left terminal node,
    the right terminal node reward is 1.

    The episode terminates at left or right endpoints.
    """

    LEFT_STEP_ACTION = 0
    RIGHT_STEP_ACTION = 1

    def __init__(self, num_nonterminal_states: int) -> None:
        super().__init__()

        assert (
            num_nonterminal_states % 2 == 1
        ), "Number of non-terminal states must be odd."

        self.num_nonterminal_states = num_nonterminal_states
        self.terminal_state_left = 0
        self.terminal_state_right = self.num_nonterminal_states + 1
        self.start_state = self.num_nonterminal_states // 2 + 1

        self.terminal_reward = 1
        self.non_terminal_reward = -1

        self.action_space: Space = Discrete(2)
        self.observation_space: Space = Discrete(self.num_nonterminal_states + 2)
        self.reward_range: Tuple[int, int] = (
            self.non_terminal_reward,
            self.terminal_reward,
        )

        self._state = None

    def __str__(self) -> str:
        return f"LineWorld{self.num_nonterminal_states}"

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value) -> None:
        assert (
            0 <= value and value <= self.num_nonterminal_states + 1
        ), "Value {value} is out of range for state."

        self._state = value

    def reset(self) -> int:
        """Set state to the middle point."""
        self.state = self.start_state

        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        if action == type(self).LEFT_STEP_ACTION:
            self.state -= 1
        elif action == type(self).RIGHT_STEP_ACTION:
            self.state += 1

        done = self.is_done()
        info = {}

        if self.state == self.terminal_state_right:
            reward = self.terminal_reward
        else:
            reward = self.non_terminal_reward

        return self.state, reward, done, info

    def is_done(self) -> bool:
        return self.state in {self.terminal_state_left, self.terminal_state_right}


class Maze2D(Env):
    UP_ACTION = 0
    DOWN_ACTION = 1
    LEFT_ACTION = 2
    RIGHT_ACTION = 3

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

        self.start_position = (0, 0)
        self.end_position = (self.height - 1, self.width - 1)

        self.blocked_positions: Set[Tuple[int, int]] = None

        self.terminal_reward = 1
        self.non_terminal_reward = -1

        self.action_space: Space = Discrete(4)
        self.observation_space: Space = Box(
            low=np.array([0, 0]),
            high=np.array([self.height, self.width]),
            dtype=np.int32,
        )
        self.reward_range: Tuple[int, int] = (
            self.non_terminal_reward,
            self.terminal_reward,
        )

        self.state: Tuple[int, int] = None

    def reset(self) -> Any:
        self.state = self.start_position

        return self.state

    def is_done(self) -> bool:
        return self.state == self.end_position

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict[str, Any]]:
        if action == type(self).LEFT_ACTION:
            self._step_left()
        elif action == type(self).RIGHT_ACTION:
            self._step_right()
        elif action == type(self).DOWN_ACTION:
            self._step_down()
        elif action == type(self).UP_ACTION:
            self._step_up()
        else:
            raise ValueError(f"Action {action} is invalid.")

        info = dict()
        done = self.is_done()

        if done:
            reward = self.terminal_reward
        else:
            reward = self.non_terminal_reward

        return self.state, reward, done, info

    def _step_right(self) -> None:
        h, w = self._get_state_height_width()
        new_h = h
        new_w = w + 1

        new_state = self._construct_state_from_height_width(new_h, new_w)
        if self._is_valid_state(new_state):
            self.state = new_state
        else:
            pass

    def _step_left(self) -> None:
        h, w = self._get_state_height_width()
        new_h = h
        new_w = w - 1

        new_state = self._construct_state_from_height_width(new_h, new_w)
        if self._is_valid_state(new_state):
            self.state = new_state
        else:
            pass

    def _step_down(self) -> None:
        h, w = self._get_state_height_width()
        new_h = h - 1
        new_w = w

        new_state = self._construct_state_from_height_width(new_h, new_w)
        if self._is_valid_state(new_state):
            self.state = new_state
        else:
            pass

    def _step_up(self) -> None:
        h, w = self._get_state_height_width()
        new_h = h + 1
        new_w = w

        new_state = self._construct_state_from_height_width(new_h, new_w)
        if self._is_valid_state(new_state):
            self.state = new_state
        else:
            pass

    def _construct_state_from_height_width(
        self, height: int, width: int
    ) -> Tuple[int, int]:
        return (height, width)

    def _get_state_height_width(self) -> Tuple[int, int]:
        return self.state

    def _is_valid_state(self, state: Tuple[int, int]) -> bool:
        return self._is_notblocked(state) and self._is_insidebounds(state)

    def _is_notblocked(self, state: Tuple[int, int]) -> bool:
        return state in self.blocked_positions

    def _is_insidebounds(self, state: Tuple[int, int]) -> bool:
        h, w = self._get_state_height_width()
        width_check = (w >= 0) and (w < self.width)
        height_check = (h >= 0) and (h < self.height)

        return width_check and height_check
