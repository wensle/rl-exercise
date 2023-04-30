import argparse
import flappy_bird_gymnasium  # noqa
import gymnasium  # noqa

# from gymnasium.utils.save_video import save_video
import pygame

from rl.models import DQN
from rl.networks import DeepMlpVersion0, Mlp


def play_with_gui(num_epoch, model: DQN) -> None:
    for _ in range(num_epoch):
        clock = pygame.time.Clock()
        episode_score = 0
        episode_reward = 0
        model.agent.reset()
        while True:
            model.env.render()

            # Getting action
            reward, done, score = model.agent.play_step(model.net, epsilon=0.0)
            episode_score += score
            episode_reward += reward

            clock.tick(60)
            if done:
                model.env.render()
                break


def play_for_validation(num_episodes, model: DQN):
    global_step_counter = 0
    for episode_counter in range(num_episodes):
        model.agent.reset()
        episode_score = 0
        episode_reward = 0
        episode_step_counter = 0
        while True:
            # Getting action
            step_reward, done, score = model.agent.play_step(model.net, epsilon=0.0)
            episode_score += score
            episode_reward += step_reward
            episode_step_counter += 1
            global_step_counter += 1

            yield (
                global_step_counter,
                episode_step_counter,
                episode_counter,
                step_reward,
                episode_score,
                episode_reward,
            )

            if done:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Load a model checkpoint and data file"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        help="Number of episodes to play",
        default=5,
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    num_episodes = args.num_episodes
    net = Mlp(obs_size=12, n_actions=2, hidden_size=512)
    target_net = Mlp(obs_size=12, n_actions=2, hidden_size=512)
    model = model = DQN.load_from_checkpoint(
        checkpoint_path, net=net, target_net=target_net
    )

    # Set the model to evaluation mode, because we don't want to train it. Then freeze
    # the parameters, because we don't want to update them.
    model.eval()
    model.freeze()

    play_with_gui(num_episodes, model)


if __name__ == "__main__":
    main()
