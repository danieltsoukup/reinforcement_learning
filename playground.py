from src.environments import Maze2D, QueueAccessControl, LineWorld
from src.policies import RandomPolicy, TabularEpsilonGreedyPolicy, ConstantPolicy
from src.experimenter import evaluate_policies
from src.control import MonteCarlo, SARSA
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


#########################
### ENVIRONMENT SETUP ###
#########################

queue_environment = QueueAccessControl(
    num_servers=4,
    customer_rewards=[5, 1],
    customer_probs=[0.5, 0.5],
    queue_size=100,
    unlock_proba=0.2,
)

lineworld_environment = LineWorld(21)

maze = Maze2D(10, 10)

block_1 = [(4, i) for i in range(0, 9)]
block_2 = [(7, i) for i in range(1, 10)]

maze.set_blocked_states(block_1 + block_2)

if __name__ == "__main__":

    environment = queue_environment

    evaluate_policies(
        environment,
        {
            "random": RandomPolicy(environment),
            "constant": ConstantPolicy(environment, environment.ACCEPT_ACTION),
        },
        10,
    )

    tabular_policy = TabularEpsilonGreedyPolicy(environment, 1)

    control = SARSA(environment, tabular_policy, 0.1, 1)

    rewards = []

    progress_bar = tqdm(range(10))
    for i in progress_bar:
        total_rewards = control.learn(10)

        mean = total_rewards.mean()
        low, high = np.quantile(total_rewards, [0.05, 0.95])
        message = f"Mean reward {mean} (90% CI: {low: .2f} to {high: .2f}, eps: {control.policy.eps: .2f})"
        progress_bar.set_description(message, refresh=True)

        # tabular_policy.eps *= 0.95

        rewards.extend(total_rewards.tolist())

    file_name = str(control) + "_" + str(environment)
    title = str(control) + " learning on " + str(environment)

    fig = environment.plot_policy(tabular_policy)
    plt.savefig("assets/test_plots/queue_test.png")

    # plt.figure(figsize=(15, 5))
    # plt.plot(np.array(rewards))
    # plt.xlabel("Episodes")
    # plt.ylabel("Total Episode Rewards")
    # plt.title(title)
    # plt.savefig(f"assets/plots/{file_name}_learning_rewards.png")

    # tabular_policy.eps = 0
    # fig = maze.plot_policy(tabular_policy)
    # fig.savefig(f"assets/plots/{file_name}_learned_policy.png")
