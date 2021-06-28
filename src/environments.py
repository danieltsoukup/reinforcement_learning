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

        Valid actions:
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
        self.current_customer = None
        self.state = None
        self.queue = None
        self.current_customer_pointer = None

    def move_pointer(self) -> int:
        """Move the queue pointer and return the next customer.

        Returns:
            int: [description]
        """
        assert (
            len(self.queue) > self.current_customer_pointer
        ), "Cannot move pointer, at the end of the queue already."

        self.current_customer_pointer += 1

        if self.current_customer_pointer < self.queue_size:
            current_customer = self.queue[self.current_customer_pointer]
        else:
            current_customer = None

        return current_customer

    def step(self, action) -> None:
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        info = {"current_customer": self.current_customer}

        next_customer = self.move_pointer()

        if action == type(self).REJECT_ACTION or self.num_free_servers == 0:
            reward = 0
            self.current_customer = next_customer

        elif action == type(self).ACCEPT_ACTION:
            reward = self.customer_rewards[self.current_customer]
            self.num_free_servers -= 1
            self.current_customer = next_customer

        self.state = np.array([self.num_free_servers, self.current_customer])

        done = self.is_done()

        self.unlock_servers()

        return self.state, reward, done, info

    def is_done(self) -> bool:
        """
        Check if the episode ended by moving past the last queue element.
        """

        return self.current_customer_pointer == self.queue_size

    def unlock_servers(self) -> None:
        """Unlock each occupied server with the given probability."""

        num_locked_servers = self.num_servers - self.num_free_servers
        num_unlock = np.random.choice(
            2, size=num_locked_servers, p=[1 - self.unlock_proba, self.unlock_proba]
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
        self.current_customer_pointer = 0
        self.num_free_servers = self.num_servers

        self.current_customer = self.move_pointer()

        self.state = self.build_state()

        return self.state

    def build_state(self) -> np.ndarray:
        return np.array([self.num_free_servers, self.current_customer])
