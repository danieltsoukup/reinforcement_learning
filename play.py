from src.environments import Maze2D, QueueAccessControl, LineWorld
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
import matplotlib.pyplot as plt


def evaluate_policies(environment: Env, policies: Dict[str, Policy], num_episodes: int):
    for name, policy in policies.items():
        experiment = Experiment(environment, policy)
        rewards = experiment.evaluate(num_episodes, n_jobs=-1)

        mean = rewards.mean()
        low, high = np.quantile(rewards, [0.05, 0.95])

        print(
            f"Policy {name} yields {mean: .2f} mean reward (90% conf. {low: .2f} to {high: .2f}).\n"
        )


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

maze = Maze2D(10, 10)

maze.add_blocked_states([(5, i) for i in range(0, 9)])

if __name__ == "__main__":

    environment = maze

    evaluate_policies(
        environment,
        {"random": RandomPolicy(environment)},
        10,
    )

    tabular_policy = TabularEpsilonGreedyPolicy(environment, 0.5)

    monte_carlo = MonteCarlo(environment, tabular_policy)

    rewards = []

    progress_bar = tqdm(range(50))
    for i in progress_bar:
        total_rewards = monte_carlo.learn(50)

        mean = total_rewards.mean()
        low, high = np.quantile(total_rewards, [0.05, 0.95])
        message = f"Mean reward {mean} (90% CI: {low: .2f} to {high: .2f}, eps: {monte_carlo.policy.eps: .2f})"
        progress_bar.set_description(message, refresh=True)

        tabular_policy.eps *= 0.95

        rewards.extend(total_rewards.tolist())

    file_name = str(monte_carlo) + "_" + str(environment)
    title = str(monte_carlo) + "learning on " + str(environment)

    plt.figure(figsize=(15, 5))
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Episode Rewards")
    plt.title(title)
    plt.savefig(f"assets/plots/{file_name}.png")
