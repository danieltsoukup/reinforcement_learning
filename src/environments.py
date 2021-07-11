from gym.core import Env
from gym.spaces import Discrete, Space, Tuple as TupleSpace
from typing import Tuple, List
import numpy as np


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
    ) -> Tuple[np.ndarray, float, bool, dict]:
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

        self.reward = 1

        self.action_space: Space = Discrete(2)
        self.observation_space: Space = Discrete(self.num_nonterminal_states + 2)
        self.reward_range: Tuple[int, int] = (0, 100)

        self._state = None

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

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Step left or right - only reward comes if the right endpoint is reached. The episode terminates at left or right endpoints."""

        if action == type(self).LEFT_STEP_ACTION:
            self.state -= 1
        elif action == type(self).RIGHT_STEP_ACTION:
            self.state += 1

        done = self.is_done()
        info = {}

        if self.state == self.terminal_state_right:
            reward = self.reward
        else:
            reward = 0

        return self.state, reward, done, info

    def is_done(self) -> bool:
        return self.state in {self.terminal_state_left, self.terminal_state_right}
