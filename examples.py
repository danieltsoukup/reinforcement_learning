from examples.ex1_maze_montecarlo_learning import run_example as run_maze_montecarlo
from examples.ex2_sarsa_hyperparameter_tuning import (
    run_example as run_sarsa_hyperparameter,
)
import sys

example = sys.argv[1]

if __name__ == "__main__":
    if example == "maze_montecarlo":
        run_maze_montecarlo()

    elif example == "sarsa_hyperparameter":
        run_sarsa_hyperparameter()

    else:
        raise NotImplementedError(f"Example {example} not recognized.")
