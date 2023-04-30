from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger

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
