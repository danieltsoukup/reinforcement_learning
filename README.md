# Reinforcement Learning Experiments

This repo explores different approaches for reinforcement learning, both policy evaluation and control problems. 

See [some examples here](EXAMPLES.md).

## Environments

The following environments are implemented.

### 2D Maze

Implemented under `Maze2D`, the agent navigates a maze with blocked cells (stepping up, down, left or right) starting at one corner (0, 0) and aiming to find the terminal state at (height-1, width-1) in the shortest amount of time. Reaching a non-terminal state is reward by -1 (terminal reward is 1).

### Line World

Implemented under `LineWorld`, a linear state space with left/right steps as actions for each state. Stepping into non-terminal states are negatively rewarded so the aim is to find the terminal state at one end-point in the shortest amount of time.


### Queue Access Control

Implemented under `QueueAccessControl`, the environment models a set of servers and a FIFO queue of customers applying for server access. At each step, we decide if the customer at the head of the queue is accepted (given an unlocked server) or rejected. Each customer has a priority and after each accepted customer, we receive a reward proportional to their priority. At each step, the in-use servers unlock at a fixed probability. The environment state holds the number of free servers and the current customer's priority. At environment reset, a fixed size queue is built by picking customers randomly according to per-priority rates. 


## Policies

The following policies are implemented.

- `ConstantPolicy`: play the same action regardless of state.
- `RandomPolicy`: select action randomly.
- `TabularGreedy` and `TabularEpsilonGreedy`: based on state-action values, select the action with highest value (or explore).

## Control methods

The following value learning and control algorithms are tested:

- `MonteCarlo`: tabular, on-policy Monte Carlo learning. We run an episode and record the state-action _returns_ i.e., the sum of rewards after the first occurrence of each state-action pair. The state-action values are the mean of the so-far observed state-action returns (updated after each episode). The simulated episode uses the an epsilon-greedy action selection based on the state-actions values calculated so far. 

- `SARSA`: tabular, on-policy SARSA learning. The state-action values are update following every step based on the rule

$$Q(s,a) \leftarrow Q(s, a) + \alpha[r + \gamma Q(s', a') - Q(s, a)]$$

where 

- $s$ is the current state and $a$ is the action selected by the $\varepsilon$-greedy policy,
- $r$ is the reward, and
- $s'$ are the resulting state and $a'$ is the next selected action.

The hyperparameters are the learning rate $\alpha$ and discount factor $\gamma$.