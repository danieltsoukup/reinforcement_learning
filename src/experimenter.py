from gym.core import Env
from src.policies import Policy
import numpy as np
from joblib import Parallel, delayed


class Experiment(object):
    def __init__(self, environment: Env, policy: Policy) -> None:
        self.environment = environment
        self.policy = policy

    def run_episode(self):
        state = self.environment.reset()
        done = False

        rewards = []
        while done is False:
            action = self.policy.select_action(state, self.environment)
            state, reward, done, info = self.environment.step(action)
            rewards.append(reward)

        return np.array(rewards)

    def evaluate(self, num_episodes: int, n_jobs: int = -1) -> float:
        """Returns the total rewards over num_episode runs."""

        reward_sequences = Parallel(n_jobs=n_jobs)(
            delayed(self.run_episode)() for _ in range(num_episodes)
        )
        total_rewards = [rewards.sum() for rewards in reward_sequences]

        return np.array(total_rewards)
