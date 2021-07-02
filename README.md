# Reinforcement Learning Experiments


## Environments

The following environments are implemented.

### Queue Access Control

Implemented under `QueueAccessControl`, the environment models a set of servers and a FIFO queue of customers applying for server access. At each step, we decide if the customer at the head of the queue is accepted (given an unlocked server) or rejected. Each customer has a priority and after each accepted customer, we receive a reward proportional to their priority. At each step, the in-use servers unlock at a fixed probability. The environment state holds the number of free servers and the current customer's priority. At environment reset, a fixed size queue is built by picking customers randomly according to per-priority rates. 

## Policies

The following policies are implemented.

- `ConstantPolicy`: play the same action regardless of state.
- `RandomPolicy`: select action randomly.