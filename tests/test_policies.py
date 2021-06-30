from src.policies import RandomPolicy
from src.experimenter import Experiment
from gym.core import Env
from gym.spaces import Discrete, Space
from typing import Tuple


class DummyEnv(Env):
    """Dummy environment for testing. There is a single action, 0 reward for 10 steps until done."""

    def __init__(self) -> None:
        super().__init__()

        self.episode_length = 10
        self.action_space: Space = Discrete(1)
        self.observation_space: Space = Discrete(1)
        self.reward_range: Tuple[int, int] = (0, 0)

        self.counter = None

    def step(self, action):
        self.counter += 1

        state = None
        reward = 0
        done = self.counter == 10
        info = None

        return state, reward, done, info

    def reset(self):
        self.counter = 0

        return None


def test_experiment():
    dummy_env = DummyEnv()

    random_policy = RandomPolicy()

    experiment = Experiment(dummy_env, random_policy)

    rewards = experiment.run_episode()

    assert len(rewards) == dummy_env.episode_length
