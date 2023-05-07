from collections import OrderedDict
from typing import List, Tuple

import gymnasium
import lightning as L
from torch import Tensor
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from rl.agents import Agent
from rl.datasets import RLDataset
from rl.losses import dqn_mse_loss
from rl.memory import ReplayBuffer
from rl.utils import weights_init


class DQN(L.LightningModule):
    """A PyTorch Lightning module implementing the Deep Q-Network (DQN) algorithm for
    reinforcement learning."""

    def __init__(
        self,
        net: nn.Module,
        target_net: nn.Module,
        batch_size: int = 32,
        lr: float = 1e-2,
        env: str = "FlappyBird-v0",
        size_hidden_layers: int = 128,
        gamma: float = 0.99,
        sync_rate: int = 1000,
        replay_size: int = 100000,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 15000,
        episode_length: int = 1000,
        warm_start_steps: int = 10000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches

            lr: learning rate

            env: gym environment tag

            gamma: discount factor

            sync_rate: how many frames do we update the target network

            replay_size: capacity of the replay buffer

            warm_start_size: how many samples do we use to fill our buffer at the start
            of training

            eps_last_frame: what frame should epsilon stop decaying

            eps_start: starting value of epsilon

            eps_end: final value of epsilon

            episode_length: max length of an episode

            warm_start_steps: max episode reward in the environment

        """
        super().__init__()
        self.save_hyperparameters(ignore=["net", "target_net"])

        self.env = gymnasium.make(self.hparams.env)

        self.net = net.apply(lambda m: weights_init(m, std_dev=0.01))
        self.target_net = target_net.apply(lambda m: weights_init(m, std_dev=0.01))
        self.dqn_mse_loss = dqn_mse_loss

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.episode = 0
        self.episode_reward = 0.0
        self.total_reward = 0.0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Fill the replay buffer with random experiences. Used to initialize the buffer
        at the start of training."""
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Compute the Q-values for a given state."""
        output = self.net(x)
        return output

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        """Compute the exploration rate as a function of the training step."""
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carry out a single training step."""

        device = self.get_device(batch)
        epsilon = self.get_epsilon(
            start=self.hparams.eps_start,
            end=self.hparams.eps_end,
            frames=self.hparams.eps_last_frame,
        )
        self.log("epsilon", epsilon)
        self.log("episode", float(self.episode))

        # step through environment with agent
        reward, done, score = self.agent.play_step(self.net, epsilon, device)
        self.log("score", float(score), prog_bar=False)
        self.episode_reward += reward
        self.log("episode_reward", float(self.episode_reward))

        # calculates training loss

        loss = self.dqn_mse_loss(batch, self.net, self.target_net, self.hparams.gamma)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            self.episode += 1

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "reward": reward,
                "train_loss": loss,
            }
        )
        self.log("total_reward", self.total_reward, prog_bar=False)

        return loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=12,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


if __name__ == "__main__":
    import flappy_bird_gymnasium  # noqa
    from lightning.pytorch.cli import LightningCLI

    default_net = {
        "class_path": "rl.networks.Mlp",
        "init_args": {"obs_size": 12, "n_actions": 2, "hidden_size": 128},
    }

    class CustomCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.set_defaults({"model.net": default_net})
            parser.set_defaults({"model.target_net": default_net})

    cli = CustomCLI(DQN)
