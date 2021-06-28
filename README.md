# Reinforcement Learning Experiments


## Environments

The following environments are implemented.

### Queue Access Control

Implemented under `QueueAccessControl`, the environments models a set of servers and a FIFO queue of customers applying for server access. At each step, we decide if the customer is accepted (given an unlocked server) or rejected. Each customer has a priority and after each accepted customer, we receive a reward proportional to their priority. We don't see the customers in the queue other than the head. At each step, the in-use servers unlock at a fixed probability.

## Policies

The following policies are implemented.