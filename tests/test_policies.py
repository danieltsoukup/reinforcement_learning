from src.policies import RandomPolicy, TabularGreedyPolicy
from src.experimenter import Experiment
from gym.core import Env
from gym.spaces import Discrete, Space
from typing import Tuple
import pytest


class DummyEnv(Env):
    """Dummy environment for testing.

    There is are two actions, two states, 0 reward for 10 steps until the episode ends."""

    def __init__(self) -> None:
        super().__init__()

        self.episode_length = 10
        self.action_space: Space = Discrete(2)
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


@pytest.fixture
def dummy_env():
    return DummyEnv()


def test_experiment(dummy_env):
    random_policy = RandomPolicy(dummy_env)

    experiment = Experiment(dummy_env, random_policy)

    rewards = experiment.run_episode()

    assert len(rewards) == dummy_env.episode_length


def test_greedy_choice(dummy_env):
    tabular_policy = TabularGreedyPolicy(dummy_env)
    tabular_policy.state_action_values[(0, 1)] = 1

    assert tabular_policy._greedy_action(0) == 1
