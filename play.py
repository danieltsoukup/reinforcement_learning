from src.environments import QueueAccessControl, LineWorld
from src.policies import (
    RandomPolicy,
    Policy,
    TabularEpsilonGreedyPolicy,
)
from src.experimenter import Experiment
from src.learning import MonteCarlo
import numpy as np
from gym.core import Env
from typing import Dict
from tqdm import tqdm


def evaluate_policies(environment: Env, policies: Dict[str, Policy], num_episodes: int):
    for name, policy in policies.items():
        experiment = Experiment(environment, policy)
        rewards = experiment.evaluate(num_episodes, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.05, 0.95])

        print(f"Policy {name} yields {mean} mean reward (90% conf. {low} to {high}).\n")


#########################
### ENVIRONMENT SETUP ###
#########################

queue_environment = QueueAccessControl(
    num_servers=4,
    customer_rewards=[100, 1],
    customer_probs=[0.05, 0.95],
    queue_size=100,
    unlock_proba=0.2,
)

lineworld_environment = LineWorld(21)

if __name__ == "__main__":

    environment = lineworld_environment

    evaluate_policies(
        environment,
        {"random": RandomPolicy(environment)},
        10,
    )

    tabular_policy = TabularEpsilonGreedyPolicy(environment, 0.5)

    monte_carlo = MonteCarlo(environment, tabular_policy)

    progress_bar = tqdm(range(100))
    for i in progress_bar:
        total_rewards = monte_carlo.learn(50)

        mean = total_rewards.mean()
        low, high = np.quantile(total_rewards, [0.05, 0.95])
        message = f"Mean reward {mean} (90% CI: {low} to {high}, eps: {round(monte_carlo.policy.eps, 2)})"
        progress_bar.set_description(message, refresh=True)

        tabular_policy.eps *= 0.95
