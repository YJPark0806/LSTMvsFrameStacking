'''
This script trains and tests both the RecurrentPPO and PPO models for the Fully Observable Environment.
'''

import pickle
import gymnasium as gym

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecMonitor

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.distributions import MultivariateNormal



#=====================================================================================================

# Custom Callback to store episode rewards and mean return
class MeanRewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MeanRewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.mean_rewards = []
        self.episode_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Add reward per step
        self.episode_reward += self.locals['rewards'][0]

        # Record and reset episode reward when episode ends
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_count += 1
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.mean_rewards.append(mean_reward)
            self.episode_reward = 0.0

        return True

#================================================================================================

if __name__ == '__main__':
    # type [python pendulum_wCV_env1.py] in your conda terminal to run this code


    # define the environment

    env_so = gym.make('Pendulum-v1')
    env_so = DummyVecEnv([lambda: env_so])

    env_mo = VecFrameStack(env_so, n_stack=4)  # Stack frames if required

    # set hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 500000


    #============================= [Model 1] Recurrent PPO with Multiple Images===========================

    # Define and Train the Model
    # Define the Model
    model_RPPOwO = RecurrentPPO('MlpLstmPolicy',
                                env_so,
                                ent_coef=0.01,
                                verbose=1,
                                gamma=0.88,
                                learning_rate=0.0001,
                                batch_size=256,
                                gae_lambda=0.95)

    # Initialize the custom callback
    mean_reward_logger_RPPOwO = MeanRewardLoggerCallback()

    # Train the model with the custom callback####################################################
    model_RPPOwO.learn(total_timesteps=total_timesteps, callback=mean_reward_logger_RPPOwO)
    #######################################################################################

    # Save the trained model
    model_RPPOwO.save("RPPOwO_model")

    with open("episode_rewards_RPPOwO.pkl", "wb") as file:  # check the path before run!
        pickle.dump(mean_reward_logger_RPPOwO.mean_rewards, file)

    # Save the total reward per episode figure - for checking purpose
    plt.figure(figsize=(10, 5))
    plt.plot(mean_reward_logger_RPPOwO.mean_rewards, label='Mean Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('<RecurrentPPO with one image>\n Mean Reward per Episode During Training')
    plt.legend()
    plt.savefig("RPPO_mean_reward_per_episode.png")

    #============================= [Model 2] PPO with Multiple Images ===========================
    # Define the Model
    model_PPOwM = PPO('MlpPolicy',
                      env_mo,  # env with multiple images
                      verbose=1,
                      device=device)

    # Initialize the custom callback
    mean_reward_logger_PPOwM = MeanRewardLoggerCallback()

    # Train the model with the custom callback
    #######################################################################################
    model_PPOwM.learn(total_timesteps=total_timesteps, callback=mean_reward_logger_PPOwM)
    #######################################################################################

    # Save the trained model, total reward per episode, and training time
    model_PPOwM.save("PPOwM_model")

    with open("episode_rewards_PPOwM.pkl", "wb") as file:  # check the path before run!
        pickle.dump(mean_reward_logger_PPOwM.mean_rewards, file)

    # Save the total reward per episode figure - for checking purpose
    plt.figure(figsize=(10, 5))
    plt.plot(mean_reward_logger_PPOwM.mean_rewards, label='Mean Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('<PPO with multiple images>\n Mean Reward per Episode During Training')
    plt.legend()
    plt.savefig("PPO_mean_reward_per_episode.png")

    #============================= Model Testing ===========================================

    # (1) Convergence Speed
    plt.figure(figsize=(10, 5))

    # Plot each series individually
    plt.plot(range(len(mean_reward_logger_RPPOwO.mean_rewards)), mean_reward_logger_RPPOwO.mean_rewards,
             label='PPO + LSTM')
    plt.plot(range(len(mean_reward_logger_PPOwM.mean_rewards)), mean_reward_logger_PPOwM.mean_rewards,
             label='PPO with 2 Images')


    # Labeling the axes and the plot
    plt.xlabel('Episodes')
    plt.ylabel('Mean Return')
    plt.title('Pendulum - Partially Observable Environment')
    plt.legend()
    plt.savefig("Partially Observable Environment.png")

    # (2) Test Results

    # Set parameters
    time_step_duration = 0.02  # Manually set the time step duration (commonly 0.02 seconds)
    success = 1
    episodes_test = 100


    # Define the function for testing
    def get_results(model, env, episodes_test=episodes_test):
        episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=episodes_test,
                                                           return_episode_rewards=True)

        # Calculate the mean reward and standard deviation of rewards
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward

    # Conduct the test for 2 models
    # Evaluate each model for custom metrics
    # Evaluate each model
    mean_reward_RPPOwO, std_reward_RPPOwO = get_results(model_RPPOwO, env_so)
    mean_reward_PPOwM, std_reward_PPOwM = get_results(model_PPOwM,env_mo)

    # Print results in tabular format
    print("\nTest Results:")
    print( f"{'Model':<10} {'Mean Reward':<20} {'Average Time (s)':<20}")
    # print(f"{'Random':<10} {mean_reward_random:.2f} +- {std_reward_random:.2f} {'':<8}")
    print( f"{'RPPOwO':<10} {mean_reward_RPPOwO:.2f} +- {std_reward_RPPOwO:.2f} {'':<8}")
    print( f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8}")


    ##################################################################################################
    # Save two tables
    # Define the output file name
    output_file = "test_and_training_results.txt"

    # Open the file in write mode
    with open(output_file, 'w') as file:
        # Write the test results table header
        print("\nTest Results:")
        print(f"{'Model':<10} {'Mean Reward':<20} {'Average Time (s)':<20}")
        # print(f"{'Random':<10} {mean_reward_random:.2f} +- {std_reward_random:.2f} {'':<8}")
        print(f"{'RPPOwO':<10} {mean_reward_RPPOwO:.2f} +- {std_reward_RPPOwO:.2f} {'':<8}")
        print(f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8}")

    print(f"Test and Training results have been saved to {output_file}")


    # ====================================================================================")