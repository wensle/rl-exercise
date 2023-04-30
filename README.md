# Reinforcement Learning Project: Flappy Bird

In this project, I am applying reinforcement learning techniques to create an agent capable of effectively playing the Flappy Bird game. The process involves the following steps:

### 1. Game Selection

I have chosen the Flappy Bird game as the focus of this project due to its suitability for reinforcement learning. The game environment is accessible and can be recreated easily.

### 2. Environment Definition

A Flappy Bird game environment was found on GitHub, which is created using the `pygame` library and uses the `gymnasium` library for compatibility with RL algorithms. I have forked the repository and made some changes to fix errors related to running the game via the command line and disabled the sound for compatibility with my system (WSL2 Ubuntu). The repository can be found [here](https://github.com/wensle/flappy-bird-gymnasium). In this notebook, we'll use the `FlappyBird-v0` environment from the `flappy_bird_gymnasium` library.

### 3. Parameter Identification

The state, action, and reward representations for the game have been defined to facilitate the RL process. The `FlappyBird-v0` environment provides state representations such as pipe positions, player's position, velocity, and rotation. The available actions are "do nothing" and "flap." Rewards are given for staying alive, successfully passing a pipe, and dying.

### 4. RL Algorithm Selection

Considering the game's simplicity and small state space, the Deep Q-Network (DQN) algorithm is an appropriate choice. I have followed tutorials to understand and implement the DQN algorithm using the PyTorch and PyTorch Lightning libraries.

### 5. Data Preprocessing

The state data will be prepared for input into the learning algorithm, which may involve feature extraction, normalization, or data augmentation.

### 6. Agent Implementation

The chosen RL algorithm, DQN, will be implemented using a relevant framework or library, ensuring the agent can interact with the game environment.

### 7. Agent Training

The RL agent will be trained to learn from its actions and received rewards, with progress and performance monitored over time.

### 8. Hyperparameter Tuning

The RL algorithm's hyperparameters will be optimized to enhance the agent's performance.

### 9. Agent Evaluation

The trained agent's performance in the game environment will be assessed against various benchmarks.

### 10. Iteration and Refinement

Based on the evaluation, the agent, reward function, or RL algorithm will be iterated and refined to improve performance.

## Reading guide

The lightning_logs folder contains the logs from the training process. The checkpoints folder contains the saved models and notebooks to analyze the results.

The rl folder contains the code for the RL agent, model, and replay buffer.
