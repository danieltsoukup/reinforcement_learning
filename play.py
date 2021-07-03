from src.environments import QueueAccessControl
from src.policies import ConstantPolicy, RandomPolicy, Policy
from src.experimenter import Experiment
from src.learning import MonteCarlo
import numpy as np
from gym.core import Env
from typing import Dict


def evaluate_policies(environment: Env, policies: Dict[str, Policy], num_episodes: int):
    for name, policy in policies.items():
        experiment = Experiment(environment, policy)
        rewards = experiment.evaluate(num_episodes, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.05, 0.95])

        print(f"Policy {name} yields {mean} mean reward (90% conf. {low} to {high}).")


if __name__ == "__main__":
    queue_environment = QueueAccessControl(
        num_servers=4,
        customer_rewards=[8, 4, 2, 1],
        customer_probs=[0.4, 0.2, 0.2, 0.2],
        queue_size=100,
        unlock_proba=0.2,
    )

    monte_carlo = MonteCarlo(queue_environment)
    experiment = Experiment(queue_environment, monte_carlo.policy)

    for _ in range(100):
        monte_carlo.learn(100)

        rewards = experiment.evaluate(10, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.05, 0.95])

        print(f"{mean} mean reward (90% conf. {low} to {high}).")
