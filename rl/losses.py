"""Loss functions for the RL models."""

from typing import Tuple

import torch
from torch import Tensor, nn


def dqn_mse_loss(
    batch: Tuple[Tensor, Tensor],
    net: nn.Module,
    target_net: nn.Module,
    gamma: float = 0.99,
) -> Tensor:
    """Calculates the mse loss using a mini batch from the replay buffer.

    Args:
        batch: current mini batch of replay data
        net: main training network
        target_net: target network of the main training network
        gamma: discount factor

    Returns:
        loss
    """
    (
        observations,
        actions,
        rewards,
        terminateds,
        truncateds,
        next_observations,
        info,
    ) = batch

    # Converts the data type of the actions tensor to torch.int64 (long). This is done
    # to ensure that the actions are represented as integer indices, which is important
    # when using the gather() function later. The gather() function expects indices to
    # be in torch.int64 format.
    #
    # The squeeze() function is used to remove dimensions of
    # size 1 from the tensor. The argument -1 passed to the function means that it will
    # remove the last dimension of size 1 from the tensor. This is done to make the
    # tensor's shape compatible with the other tensors involved in the computation, such
    # as when calling the gather() function.
    actions = actions.long().squeeze(-1)

    # Obtain the Q-values of the chosen actions for each state in the mini-batch.
    #
    # Computes the Q-values for all possible actions given the input states using the
    # neural network model net. The output is a tensor of shape (batch_size,
    # num_actions) where each row contains the Q-values for all possible actions in a
    # particular state.
    q_values_all_actions = net(observations)

    # The actions tensor is of shape (batch_size,), and it needs to be converted into a
    # 2D tensor to be compatible with the output of net(states). By calling
    # unsqueeze(-1), we add an extra dimension at the end, resulting in a shape of
    # (batch_size, 1).
    actions_unsqueezed = actions.unsqueeze(-1)

    # The gather() function is used to select one Q-value per state based on the
    # actions_unsqueezed tensor. The 1 argument indicates that the gathering is done
    # along the second dimension (i.e., across actions). For each row (state) in the
    # q_values_all_actions tensor, the gather() function picks the Q-value corresponding
    # to the action specified in the actions_unsqueezed tensor. The output tensor will
    # have the same shape as the actions_unsqueezed tensor, which is (batch_size, 1)
    # after the unsqueezing operation.
    q_values_selected_actions = q_values_all_actions.gather(1, actions_unsqueezed)

    # The squeeze() function is used to remove the last dimension of size 1 from the
    # tensor, converting it back to a shape of (batch_size,). This tensor now contains
    # the Q-values for the chosen actions in each state.
    q_values_chosen_actions = q_values_selected_actions.squeeze(-1)

    with torch.no_grad():
        # Compute the Q-values for all possible actions given the next states
        # using the target network target_net. The output is a tensor of shape
        # (batch_size, num_actions) where each row contains the Q-values for all
        # possible actions in a particular next state.
        next_state_q_values = target_net(next_observations).squeeze(0)

        # The max() function is used to find the maximum Q-value along the second
        # dimension (i.e., across actions) for each next state. This corresponds to the
        # action with the highest Q-value in each next state. The output is a tuple,
        # where the first element is a tensor of shape (batch_size,) containing the
        # maximum Q-values, and the second element contains their indices (actions). We
        # only need the maximum Q-values, so we store them in the next_state_values
        # variable and ignore the indices using _.
        next_state_values, _ = next_state_q_values.max(1)

        # The dones tensor is a boolean tensor indicating whether each next state is a
        # terminal state (episode ended) or not. For terminal states, there is no future
        # reward, so the Q-value should be 0. This line sets the Q-values for terminal
        # states in the next_state_values tensor to 0.
        next_state_values[terminateds | truncateds] = 0.0

        # The detach() function is called on the next_state_values tensor to create a
        # new tensor with the same content but without the gradient computation history.
        # This is done to prevent gradients from flowing through the target network
        # during the backpropagation step, as the target network's parameters should not
        # be updated during the learning process.
        next_state_values = next_state_values.detach()

    # The expected_state_action_values tensor is calculated by multiplying the
    # next_state_values tensor by the discount factor gamma and adding the immediate
    # rewards. This tensor represents the target Q-values that the DQN algorithm aims to
    # achieve for the current state-action pairs.
    expected_state_action_values = next_state_values * gamma + rewards

    return nn.MSELoss()(q_values_chosen_actions, expected_state_action_values)
