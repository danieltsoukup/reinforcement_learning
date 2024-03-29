import pytest
from src.environments import QueueAccessControl, Maze2D
from src.policies import RandomPolicy, TabularEpsilonGreedyPolicy
import pandas as pd


@pytest.fixture
def customer_rewards():

    return [5, 1]


@pytest.fixture
def queue_env(customer_rewards):
    env = QueueAccessControl(
        num_servers=4,
        customer_rewards=customer_rewards,
        customer_probs=[0.4, 0.6],
        queue_size=100,
        unlock_proba=0.2,
    )

    return env


def test_queueenv_actionspace(queue_env):

    assert queue_env.action_space.n == 2


def test_queueenv_reset_statecheck(queue_env):
    state = queue_env.reset()
    state_shape_check = state.shape == (2,)
    server_num_check = state[0] == queue_env.num_servers
    customer_check = state[1] in range(queue_env.num_customer_types)

    assert state_shape_check and server_num_check and customer_check


def test_queueenv_reset_pointer_check(queue_env):
    _ = queue_env.reset()

    assert queue_env.current_customer_pointer == 0


def test_step_queue(queue_env):
    _ = queue_env.reset()
    state, reward, done, info = queue_env.step(0)

    assert queue_env.current_customer_pointer == 1


def test_step_queue_reward(queue_env, customer_rewards):
    _ = queue_env.reset()
    state, reward, done, info = queue_env.step(1)

    assert reward == customer_rewards[info["current_customer"]]


def test_step_done(queue_env):
    _ = queue_env.reset()

    for _ in range(queue_env.queue_size):
        state, reward, done, info = queue_env.step(0)

    assert done


def test_step_not_done(queue_env):
    _ = queue_env.reset()

    for _ in range(queue_env.queue_size - 1):
        state, reward, done, info = queue_env.step(0)

    assert not done


def test_lock_all(queue_env):
    _ = queue_env.reset()

    for _ in range(queue_env.num_servers):
        _ = queue_env.step(1, unlock=False)

    assert queue_env.num_free_servers == 0


def test_lock_one(queue_env):
    _ = queue_env.reset()

    _ = queue_env.step(1, unlock=False)

    assert queue_env.num_free_servers == queue_env.num_servers - 1


def test_unlock_one(queue_env):
    _ = queue_env.reset()
    _ = queue_env.step(1, unlock=False)

    queue_env.unlock_servers(1)

    assert queue_env.num_free_servers == queue_env.num_servers


def test_unlock_zero_proba(queue_env):
    _ = queue_env.reset()
    _ = queue_env.step(1, unlock=False)

    queue_env.unlock_servers(0)

    assert queue_env.num_free_servers == queue_env.num_servers - 1


def test_action_selection(queue_env):
    policy = TabularEpsilonGreedyPolicy(queue_env, eps=1)
    state = queue_env.reset()
    record = pd.DataFrame(columns=["state", "action", "reward"], index=range(100))
    for i in range(100):
        action = policy.select_action(state)
        record.loc[i, "state"] = state
        record.loc[i, "action"] = action
        state, reward, _, _ = queue_env.step(action)
        record.loc[i, "reward"] = reward

    record.to_csv("assets/test_plots/queue_game_play.csv")


############
### MAZE ###
############


@pytest.fixture
def maze() -> Maze2D:
    maze = Maze2D(5, 5)

    return maze


def test_maze_reset(maze):
    state = maze.reset()

    assert state == (0, 0)


def test_maze_step_up(maze):
    _ = maze.reset()
    maze._step_up()

    assert maze.state == (0, 1)


def test_maze_step_right(maze):
    _ = maze.reset()
    maze._step_right()

    assert maze.state == (1, 0)


def test_maze_step_left_fail(maze):
    _ = maze.reset()
    maze._step_left()

    assert maze.state == (0, 0)


def test_maze_step_down_fail(maze):
    _ = maze.reset()
    maze._step_down()

    assert maze.state == (0, 0)


def test_maze_block(maze):
    _ = maze.reset()

    maze.set_blocked_states([(1, 1)])

    assert maze._is_notblocked((1, 1)) is False


def test_maze_plot(maze):
    _ = maze.reset()
    # maze.set_blocked_states([(2, 2), (1, 1), (1, 2)])
    # maze._step_up()

    fig = maze.plot()
    fig.savefig("assets/test_plots/maze_plot_test.png")


def test_maze_policy_plot(maze):
    _ = maze.reset()

    maze.set_blocked_states([(2, 2), (1, 1), (1, 2), (1, 3), (2, 4)])
    maze._step_up()

    policy = RandomPolicy(maze)

    fig = maze.plot_policy(policy)
    fig.savefig("assets/test_plots/maze_plot_policy_test.png")
