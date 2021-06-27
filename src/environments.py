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

        self.action_space: Space = Discrete(2)
        self.observation_space: Space = TupleSpace(
            (Discrete(self.num_servers + 1), Discrete(self.num_customer_types))
        )
        self.reward_range: Tuple[int, int] = (
            min(self.customer_rewards),
            max(customer_rewards),
        )

        self.state = None
        self.queue = None

    def get_current_customer(self) -> int:
        return self.state[1]

    def pop_queue(self) -> int:
        next_customer, self.queue = self.queue[0], self.queue[1:]

        return next_customer

    def get_num_free_servers(self) -> int:
        return self.state[0]

    def step(self, action) -> None:
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        num_free_servers = self.get_num_free_servers()
        current_customer = self.get_current_customer()
        next_customer = self.pop_queue()

        if action == type(self).REJECT_ACTION or num_free_servers == 0:
            reward = 0
            self.state = np.array([num_free_servers, next_customer])
        elif action == type(self).ACCEPT_ACTION:
            reward = self.customer_rewards[current_customer]
            self.state = np.array([num_free_servers - 1, next_customer])

        done = len(self.queue) == 0
        info = {}

        return self.state, reward, done, info

    def reset(self) -> np.ndarray:
        self.queue = np.random.choice(
            self.num_customer_types, p=self.customer_probs, size=self.queue_size
        )

        customer = self.pop_queue()
        self.state = np.array([self.num_servers, customer])

        return self.state
