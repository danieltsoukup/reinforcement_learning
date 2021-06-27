from gym.core import Env
from gym.spaces import Discrete, Space, Tuple as TupleSpace
from typing import Tuple, List
import numpy as np


class QueueAccessControl(Env):
    def __init__(
        self, num_servers: int, customer_rewards: List[int], customer_probs: List[float]
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

        self.action_space: Space = Discrete(2)
        self.observation_space: Space = TupleSpace(
            (Discrete(self.num_servers + 1), Discrete(self.num_customer_types))
        )
        self.reward_range: Tuple[int, int] = (
            min(self.customer_rewards),
            max(customer_rewards),
        )

        self.state = None

    def step(self, action) -> None:
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

    def reset(self) -> np.ndarray:
        customer = np.random.choice(self.num_customer_types, p=self.customer_probs)
        state = np.array([self.num_servers, customer])

        return state
