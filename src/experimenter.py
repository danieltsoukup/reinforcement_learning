from gym.core import Env
from src.policies import Policy
import numpy as np
from joblib import Parallel, delayed
from typing import Dict


def evaluate_policies(environment: Env, policies: Dict[str, Policy], num_episodes: int):
    """
    Utility function to evaluate multiple policies on a single environment.
    """
    for name, policy in policies.items():
        experiment = Experiment(environment, policy)
        rewards = experiment.evaluate(num_episodes, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.05, 0.95])

        print(
            f"Policy {name} yields {mean: .2f} mean reward (90% CI: {low: .2f} to {high: .2f}).\n"
        )


class Experiment(object):
    """
    Run multiple episodes using an environment-policy pair, returning the array of total rewards per episode.
    """

    def __init__(self, environment: Env, policy: Policy) -> None:
        self.environment = environment
        self.policy = policy

    def run_episode(self) -> np.ndarray:
        state = self.environment.reset()
        done = False

        rewards = []
        while done is False:
            action = self.policy.select_action(state)
            state, reward, done, info = self.environment.step(action)
            rewards.append(reward)

        return np.array(rewards)

    def evaluate(self, num_episodes: int, n_jobs: int = -1) -> np.ndarray:
        """Returns the total rewards over num_episode runs."""

        reward_sequences = Parallel(n_jobs=n_jobs)(
            delayed(self.run_episode)() for _ in range(num_episodes)
        )
        total_rewards = [rewards.sum() for rewards in reward_sequences]

        return np.array(total_rewards)
