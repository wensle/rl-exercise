from torch import nn


class Mlp(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        output = self.net(x.float())

        return output


class DeepMlpVersion0(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        size_layer_1 = hidden_size
        size_layer_2 = int(hidden_size / 2)
        size_layer_3 = int(hidden_size / 4)
        self.net = nn.Sequential(
            nn.Linear(obs_size, size_layer_1),
            nn.ReLU(),
            nn.Linear(size_layer_1, size_layer_2),
            nn.ReLU(),
            nn.Linear(size_layer_2, size_layer_3),
            nn.ReLU(),
            nn.Linear(size_layer_3, n_actions),
        )

    def forward(self, x):
        output = self.net(x.float())

        return output
