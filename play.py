from src import environments
from src.environments import QueueAccessControl, LineWorld
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

lineworld_environment = LineWorld(5)

if __name__ == "__main__":

    environment = lineworld_environment

    evaluate_policies(
        environment,
        {"random policy": RandomPolicy(environment)},
        10,
    )

    monte_carlo = MonteCarlo(environment)

    for i in range(100):
        total_rewards = monte_carlo.learn(50)

        mean = total_rewards.mean()
        low, high = np.quantile(total_rewards, [0.05, 0.95])

        print(
            f"{i} -- MC {mean} mean reward (90% conf. {low} to {high}) (eps {round(monte_carlo.policy.eps, 2)})."
        )

        monte_carlo.policy.eps *= 0.95

        # if i % 10 == 0:

        #     old_eps = monte_carlo.policy.eps

        #     monte_carlo.policy.eps = 0

        #     evaluate_policies(
        #         queue_environment,
        #         {f"monte carlo with eps {monte_carlo.policy.eps}": monte_carlo.policy},
        #         10,
        #     )

        #     monte_carlo.policy.eps = old_eps
