from typing import Tuple

import gymnasium
import numpy as np
import torch
from torch import nn

from rl.memory import Experience, ReplayBuffer
from rl.utils import env_step_experience_adapter
import random


class Agent:
    """An agent class for interacting with a Gymnasium environment and storing
    experiences in a replay buffer."""

    def __init__(self, env: gymnasium.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Initializes the Agent with a Gymnasium environment and a replay buffer.

        Args:
            env: The Gymnasium environment to interact with.

            replay_buffer: A replay buffer to store experiences.
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.state: np.ndarray
        self.score: float
        self.reset()

    def reset(self) -> None:
        """Resets the environment and updates the agent's state and score."""
        self.state, info = self.env.reset(seed=random.randint(0, 1e9))
        self.score = info["score"]

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Determines the action to carry out using an epsilon-greedy policy with the given
        network.

        Args:
            net: The DQN network to use for action selection.

            epsilon: The probability of taking a random action instead of the optimal
            action.

            device: The device to use for computation.

        Returns:
            The selected action.
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(self.state.reshape(1, -1))

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> Tuple[float, bool, float]:
        """
        Performs a single interaction step between the agent and the environment, and
        stores the experience in the replay buffer.

        Args:
            net: The DQN network to use for action selection.

            epsilon: The probability of taking a random action instead of the optimal
            action.

            device: The device to use for computation.

        Returns:
            A tuple of the reward and a boolean indicating whether the episode is over.
        """

        action: int = self.get_action(net, epsilon, device)

        experience: Experience = env_step_experience_adapter(
            self.env,
            self.state,
            action,
        )

        self.replay_buffer.append(experience)

        self.state = experience.next_observation
        self.score = experience.score

        # if experience.terminated or experience.truncated:
        #     self.reset()

        return (
            experience.reward,
            experience.terminated or experience.truncated,
            experience.score,
        )
