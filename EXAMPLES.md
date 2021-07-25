# Examples

## Monte Carlo control and Maze2D

We used the classical Monte Carlo method to learn state-action values for a 2D maze environment. The learning was on-policy, we used an epsilon-greedy policy based on the current state-action values, exponentially decaying epsilon as learning progressed to decrease exploration. 

The negative rewards per episode are plotted:

![reward line plot](assets/plots/MonteCarlo_Maze2D_10x10_learning_rewards.png)

The final, near optimal, action selection by the greedy policy is shown below:

![maze policy](assets/plots/MonteCarlo_Maze2D_10x10_learned_policy.png)

The experiment can be reproduced by running:

```bash
python examples.py maze_montecarlo
```

Edit the file to change the maze setup or learning parameters.

## Tuning SARSA hyperparameters with Optuna

The goal of this experiment was to tune the `alpha` and `gamma` parameters for SARSA on Maze2D. We optimized for KPI that incorporates both the final greedy policy reward and the speed of learning for the final policy. For efficiency, we limit terminate the episodes (if necessary) after 100 steps (the optimal policy ends in 34 steps).

The plot below shows the contour plot for the tested hyperparameters:

![HP tuning contour](assets/plots/sarsa_tuning_contour.png)

The following two figures show the rewards during learning and final policy for the best parameters:

![reward line plot](assets/plots/SarsaControl_alpha0.50_gamma0.95_Maze2D_10x10_learning_rewards.png)

It is evident that SARSA is much faster than MC in learning and the final policy is overall better:

![best policy](assets/plots/SarsaControl_alpha0.50_gamma0.95_Maze2D_10x10_learned_policy.png)


The experiment can be reproduced by running:

```bash
python examples.py sarsa_hyperparameter
```
