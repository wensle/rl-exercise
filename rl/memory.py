import collections
from typing import NamedTuple, Tuple
import numpy as np


class Experience(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    terminated: bool
    truncated: bool
    next_observation: np.ndarray
    score: float


class ReplayBuffer:
    """A replay buffer for storing and sampling past experiences for training an RL
    agent."""

    def __init__(self, capacity: int) -> None:
        """
        Initializes the replay buffer with the given capacity.

        Args:
            capacity: The maximum number of experiences to store in the buffer.
        """
        self._buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        """
        Returns the current number of experiences in the buffer.

        Returns:
            The number of experiences currently in the buffer.
        """
        return len(self._buffer)

    def append(self, experience: Experience) -> None:
        """
        Adds an experience to the buffer.

        Args:
            experience: A tuple (observation, action, reward, terminated, truncated,
            next_observation, score).
        """
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        """Samples a batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A tuple of arrays containing the sampled experiences: Observations, Actions,
            Rewards, Terminated flags (boolean), Truncated flags (boolean), Next
            observations, Scores
        """
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        (
            observations,
            actions,
            rewards,
            terminateds,
            truncateds,
            next_observations,
            scores,
        ) = zip(*[self._buffer[idx] for idx in indices])

        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            np.array(next_observations),
            np.array(scores, dtype=np.float32),
        )


if __name__ == "__main__":
    # Create a buffer
    buffer = ReplayBuffer(capacity=100)

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
        buffer.append(experience)

    # Sample a batch
    batch = buffer.sample(batch_size=5)

    # Print the shape for each element in the batch
    for element in batch:
        print(element.shape)
