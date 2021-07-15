from src.environments import Maze2D, QueueAccessControl, LineWorld
from src.policies import (
    RandomPolicy,
    TabularEpsilonGreedyPolicy,
)
from src.experimenter import evaluate_policies
from src.learning import MonteCarlo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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

block_1 = [(4, i) for i in range(0, 9)]
block_2 = [(7, i) for i in range(1, 10)]

maze.set_blocked_states(block_1 + block_2)

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

    progress_bar = tqdm(range(200))
    for i in progress_bar:
        total_rewards = monte_carlo.learn(1)

        mean = total_rewards.mean()
        low, high = np.quantile(total_rewards, [0.05, 0.95])
        message = f"Mean reward {mean} (90% CI: {low: .2f} to {high: .2f}, eps: {monte_carlo.policy.eps: .2f})"
        progress_bar.set_description(message, refresh=True)

        tabular_policy.eps *= 0.97

        rewards.extend(total_rewards.tolist())

        # if i == 100:
        #     maze.set_blocked_states(block_1)

    file_name = str(monte_carlo) + "_" + str(environment)
    title = str(monte_carlo) + "learning on " + str(environment)

    plt.figure(figsize=(15, 5))
    plt.plot(-1 * np.array(rewards))
    plt.xlabel("Episodes")
    plt.ylabel("Total Episode Negative Rewards (log scale)")
    plt.yscale("log")
    plt.title(title)
    plt.savefig(f"assets/plots/{file_name}.png")
