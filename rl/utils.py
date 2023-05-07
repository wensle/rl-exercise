from torch import nn
from rl.memory import Experience


def env_step_experience_adapter(env, observation, action) -> Experience:
    next_observation, reward, terminated, truncated, info = env.step(action)

    reward = float(reward)
    score = float(info["score"])

    experience = Experience(
        observation,
        action,
        reward,
        terminated,
        truncated,
        next_observation,
        score,
    )

    return experience


def weights_init(m, std_dev=0.01):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, std_dev)
        m.bias.data.fill_(0)
