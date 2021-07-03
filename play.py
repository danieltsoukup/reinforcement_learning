from src.environments import QueueAccessControl
from src.policies import ConstantPolicy, RandomPolicy, Policy
from src.experimenter import Experiment
from src.learning import MonteCarlo
import numpy as np
from gym.core import Env
from typing import List

environment = QueueAccessControl(
    num_servers=4,
    customer_rewards=[8, 4, 2, 1],
    customer_probs=[0.4, 0.2, 0.2, 0.2],
    queue_size=100,
    unlock_proba=0.5,
)


policies = {
    "always_reject": ConstantPolicy(environment, environment.REJECT_ACTION),
    "always_accept": ConstantPolicy(environment, environment.ACCEPT_ACTION),
    "random_action": RandomPolicy(environment),
}


def evaluate_policies(environment: Env, policies: List[Policy], num_episodes: int):
    for name, policy in policies.items():
        experiment = Experiment(environment, policy)
        rewards = experiment.evaluate(num_episodes, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.5, 0.95])

        print(f"Policy {name} yields {mean} mean reward (90% conf. {low} to {high}).")


if __name__ == "__main__":
    monte_carlo = MonteCarlo(environment)
    experiment = Experiment(environment, monte_carlo.policy)

    for _ in range(100):
        monte_carlo.learn(100)

        rewards = experiment.evaluate(10, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.5, 0.95])

        print(f"{mean} mean reward (90% conf. {low} to {high}).")
