import optuna
from src.environments import Maze2D
from src.policies import (
    TabularEpsilonGreedyPolicy,
)
from src.experimenter import evaluate_policies
from src.control import SARSA


def sarsa_objective(trial, environment) -> float:
    alpha_low, alpha_high = 0.1, 0.5
    alpha = trial.suggest_uniform("alpha", alpha_low, alpha_high)

    gamma_low, gamma_high = 0.1, 0.9
    gamma = trial.suggest_uniform("gamma", gamma_low, gamma_high)

    # train SARSA with eps-greedy and return the greedy total reward
    tabular_policy = TabularEpsilonGreedyPolicy(environment, 0.5)

    _ = SARSA(environment, tabular_policy, alpha, gamma)

    pass


def setup_maze() -> Maze2D:
    maze = Maze2D(10, 10)

    block_1 = [(4, i) for i in range(0, 9)]
    block_2 = [(7, i) for i in range(1, 10)]
    maze.set_blocked_states(block_1 + block_2)

    return maze


if __name__ == "__main__":
    maze = setup_maze()

    n_trials = 100
    study = optuna.create_study()
    study.optimize(lambda trial: sarsa_objective(trial, maze), n_trials=n_trials)

    print(study.best_params)
