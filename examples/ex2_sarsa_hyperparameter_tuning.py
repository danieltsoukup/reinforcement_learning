import optuna
from optuna.visualization import plot_contour
from src.environments import Maze2D
from src.policies import (
    TabularEpsilonGreedyPolicy,
)
from src.experimenter import Experiment
from src.control import SARSA
import numpy as np
import matplotlib.pyplot as plt

alpha_low, alpha_high = 0.1, 0.5
gamma_low, gamma_high = 0.7, 0.99
epsilon = 0.75
epsilon_decay = 0.95


def sarsa_objective(trial, environment) -> float:
    """
    Run a SARSA learning with sampled alpha and gamma and
    evaluate. The KPI is the mean of the final reward and the mean of the rewards during learning
    to penalize slow learning as well.
    """

    alpha = trial.suggest_uniform("alpha", alpha_low, alpha_high)

    gamma = trial.suggest_uniform("gamma", gamma_low, gamma_high)

    # setup
    tabular_policy = TabularEpsilonGreedyPolicy(environment, epsilon)
    control = SARSA(environment, tabular_policy, alpha, gamma)

    # learn
    learning_rewards = []
    for _ in range(50):
        rewards = control.learn(10, episode_limit=100)
        tabular_policy.eps *= epsilon_decay

        learning_rewards.extend(rewards.tolist())

    # evaluate
    tabular_policy.eps = 0
    experimenter = Experiment(environment, tabular_policy, step_limit=100)
    final_reward = experimenter.evaluate(1).mean()

    learning_mean_reward = np.mean(learning_rewards)

    kpi = 0.5 * final_reward + 0.5 * learning_mean_reward

    return kpi


def setup_maze() -> Maze2D:
    maze = Maze2D(10, 10)

    block_1 = [(4, i) for i in range(0, 9)]
    block_2 = [(7, i) for i in range(1, 10)]
    maze.set_blocked_states(block_1 + block_2)

    return maze


def run_example():
    maze = setup_maze()

    n_trials = 50
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: sarsa_objective(trial, maze), n_trials=n_trials)

    fig = plot_contour(study)
    fig.write_html("assets/plots/sarsa_tuning_contour.html")
    fig.write_image("assets/plots/sarsa_tuning_contour.png")

    # retrain with the best setup
    alpha = study.best_params["alpha"]
    gamma = study.best_params["gamma"]

    maze = setup_maze()
    tabular_policy = TabularEpsilonGreedyPolicy(maze, epsilon)
    control = SARSA(maze, tabular_policy, alpha, gamma)

    learning_rewards = []
    for _ in range(50):
        rewards = control.learn(10, episode_limit=100)
        tabular_policy.eps *= epsilon_decay

        learning_rewards.extend(rewards.tolist())

    file_name = str(control) + "_" + str(maze)
    title = str(control) + " learning on " + str(maze)

    plt.figure(figsize=(15, 5))
    plt.plot(np.array(learning_rewards))
    plt.xlabel("Episodes")
    plt.ylabel("Total Episode Rewards")
    plt.title(title)
    plt.savefig(f"assets/plots/{file_name}_learning_rewards.png")

    tabular_policy.eps = 0
    fig = maze.plot_policy(tabular_policy)
    fig.savefig(f"assets/plots/{file_name}_learned_policy.png")
