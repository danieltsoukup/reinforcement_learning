from src.environments import QueueAccessControl
from src.policies import ConstantPolicy, RandomPolicy, Policy
from src.experimenter import Experiment
from src.learning import MonteCarlo
import numpy as np
from gym.core import Env
from typing import List

environment = QueueAccessControl(
    num_servers=10,
    customer_rewards=[100, 1],
    customer_probs=[0.1, 0.9],
    queue_size=100,
    unlock_proba=0.5,
)


policies = {
    "always_reject": ConstantPolicy(environment.REJECT_ACTION),
    "always_accept": ConstantPolicy(environment.ACCEPT_ACTION),
    "random_action": RandomPolicy(),
}

num_episodes = 1000


def evaluate_policies(environment: Env, policies: List[Policy], num_episodes: int):
    for name, policy in policies.items():
        experiment = Experiment(environment, policy)
        rewards = experiment.evaluate(num_episodes, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.5, 0.95])

        print(f"Policy {name} yields {mean} mean reward (90% conf. {low} to {high}).")


if __name__ == "__main__":
    monte_carlo = MonteCarlo(environment)
    monte_carlo.learn(100)

    evaluate_policies(environment, {"monte_carlo": monte_carlo.policy}, 100)
