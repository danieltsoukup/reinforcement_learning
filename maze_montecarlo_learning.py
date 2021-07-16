from src.environments import Maze2D
from src.policies import (
    TabularEpsilonGreedyPolicy,
)
from src.learning import MonteCarlo
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == "__main__":

    ### SETUP ###

    # environment setup
    maze_environment = Maze2D(10, 10)

    block_1 = [(4, i) for i in range(0, 9)]
    block_2 = [(7, i) for i in range(1, 10)]
    maze_environment.set_blocked_states(block_1 + block_2)

    # policy setup
    tabular_policy = TabularEpsilonGreedyPolicy(maze_environment, 0.5)

    # learning setup
    monte_carlo = MonteCarlo(maze_environment, tabular_policy)

    ### LEARNING ###

    rewards = []
    progress_bar = tqdm(range(100))
    for i in progress_bar:
        total_rewards = monte_carlo.learn(50)

        # print performance
        mean = total_rewards.mean()
        low, high = np.quantile(total_rewards, [0.05, 0.95])
        message = f"Mean reward {mean} (90% CI: {low: .2f} to {high: .2f}, eps: {monte_carlo.policy.eps: .2f})"
        progress_bar.set_description(message, refresh=True)

        # decay exploration rate
        tabular_policy.eps *= 0.99

        # save rewards
        rewards.extend(total_rewards.tolist())

    ### FIGURES ###

    file_name = str(monte_carlo) + "_" + str(maze_environment)
    title = str(monte_carlo) + " learning on " + str(maze_environment)

    plt.figure(figsize=(15, 5))
    plt.plot(-1 * np.array(rewards))
    plt.xlabel("Episodes")
    plt.ylabel("Total Episode Negative Rewards (log scale)")
    plt.yscale("log")
    plt.title(title)
    plt.savefig(f"assets/plots/{file_name}_learning_rewards.png")

    tabular_policy.eps = 0
    fig = maze_environment.plot_policy(tabular_policy)
    fig.savefig(f"assets/plots/{file_name}_learned_policy.png")
