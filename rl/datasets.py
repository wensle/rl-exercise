from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from rl.memory import Experience, ReplayBuffer


class RLDataset(IterableDataset):
    """PyTorch IterableDataset that samples batches of experiences from a replay
    buffer."""

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        """
        Initializes the dataset with a replay buffer and the desired sample size.

        Args:
            buffer: The replay buffer to sample experiences from.
            sample_size: The number of experiences to sample per batch.
        """
        self.buffer: ReplayBuffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        """
        Returns an iterator that samples batches of experiences from the replay buffer.

        Returns:
            A tuple of arrays containing the sampled experiences: Observations, Actions,
            Rewards, Terminated flags (boolean), Truncated flags (boolean), Next
            observations, Scores
        """
        (
            obversations,
            actions,
            rewards,
            terminateds,
            truncateds,
            next_observations,
            scores,
        ) = self.buffer.sample(self.sample_size)

        for i in range(terminateds.shape[0]):
            yield (
                obversations[i],
                actions[i],
                rewards[i],
                terminateds[i],
                truncateds[i],
                next_observations[i],
                scores[i],
            )


if __name__ == "__main__":
    # Create a buffer
    _buffer = ReplayBuffer(capacity=100)

    # Add some experiences
    for i in range(10):
        experience = Experience(
            observation=np.random.random((10, 10)),
            action=np.random.randint(0, 10),
            reward=np.random.random(),
            terminated=np.random.random() > 0.5,
            truncated=np.random.random() > 0.5,
            next_observation=np.random.random((10, 10)),
            score=np.random.randint(0, 100),
        )
        _buffer.append(experience)

    # Create a dataset
    dataset = RLDataset(buffer=_buffer, sample_size=5)

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=2)

    # Iterate over the dataloader
    for batch in dataloader:
        print(batch)
        break
