import pytest
from src.environments import QueueAccessControl


@pytest.fixture
def queue_env():
    env = QueueAccessControl(
        num_servers=4,
        customer_rewards=[1, 2, 4, 8],
        customer_probs=[0.4, 0.2, 0.2, 0.2],
        queue_size=100,
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


def test_queueenv_reset_queuecheck(queue_env):
    _ = queue_env.reset()
    queue_length_check = len(queue_env.queue) == queue_env.queue_size - 1

    assert queue_length_check


def test_step_queue(queue_env):
    _ = queue_env.reset()
    state, reward, done, info = queue_env.step(0)

    assert len(queue_env.queue) == queue_env.queue_size - 2
