'''
This script is used to test the model trained using the 'pendulum_wCV_RPPO_training' file.

Please ensure that the model name is aligned with the one defined during training.
'''

import pickle
import gym
from gym import spaces
from collections import deque
from gym.spaces import Box
import time

import cv2
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
# Here, we define the environment wrapper

# Env with single image
# Note : For env with multiple images input, we wrap again with VecFrameStack
class ImageObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=36, height=36):
        super(ImageObservationWrapper, self).__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):

        img = self.env.render(mode='rgb_array')  # Capture the rendered image from the environment

        # Crop the image to focus on the pendulum
        # assuming the pendulum is centered in the middle
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        crop_size = 250
        img = img[center_y - crop_size//2:center_y + crop_size//2, center_x - crop_size//2:center_x + crop_size//2]

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = cv2.resize(img, (self.width, self.height))  # Resize the image to the desired size
        img = img  # NO normalization

        # Add a channel dimension to the image (from (height, width) to (height, width, 1)), to make it compatible with CnnPolciy
        img = img[:, :, None]

        # # Plot the image
        #   # NOTE THAT When you display a grayscale image using imshow,
        #   # Matplotlib uses a colormap to map the single-channel grayscale values to colors
        # plt.imshow(img)
        # plt.axis('off')  # Turn off the axis labels
        # plt.show(block=False)  # Non-blocking show
        # plt.pause(0.001)  # Pause to allow the plot to update

        return img

    def reset(self, **kwargs):
        # Reset the environment and return both the observation and an empty info dict
        obs = self.env.reset(**kwargs)
        img_obs = self.observation(obs)
        return img_obs, {}  # Return the observation and an empty info dictionary

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        img_obs = self.observation(obs)
        return img_obs, reward, done, done, info  # Return observation, reward, terminated, truncated, info

    def render(self, mode='human', **kwargs):
        return self.env.render(mode=mode, **kwargs)

#================================================================================================

if __name__ == '__main__':
    # type [python pendulum_wCV_RPPO_testing.py] in your conda terminal to run this code


    # define the environment

    # observation with single image
    env_si = gym.make('Pendulum-v1')
    env_si = ImageObservationWrapper(env_si)  # Creates an object of ImageInputWrapper class; Inputs the screen image
    env_si = DummyVecEnv([lambda: env_si])  # returns the env object; necessary in stablebaseline3

    # # observation with multiple images
    # env_mi = VecFrameStack(env_si, n_stack=2)  # n_stack is 2

    # set hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model_RPPOwO = PPO.load("models/RPPOwO_checkpoint_10000.zip", env=env_si, device='cuda')  # check the name

    # fetch the episode rewards
    # with open("logs/RPPOwO_mean_rewards_10000.pkl", "rb") as file:  # Check the name
    #     episode_rewards_RPPOwO_4 = pickle.load(file)

    #===============================================================================================================
    with open("logs/RPPOwO_mean_rewards_2500.pkl", "rb") as file:  # Check the name
        episode_rewards_RPPOwO_1 = pickle.load(file)
    with open("logs/RPPOwO_mean_rewards_5000.pkl", "rb") as file:  # Check the name
        episode_rewards_RPPOwO_2 = pickle.load(file)
    with open("logs/RPPOwO_mean_rewards_7500.pkl", "rb") as file:  # Check the name
        episode_rewards_RPPOwO_3 = pickle.load(file)
    with open("logs/RPPOwO_mean_rewards_10000.pkl", "rb") as file:  # Check the name
        episode_rewards_RPPOwO_4 = pickle.load(file)

    print(len(episode_rewards_RPPOwO_1))
    print(len(episode_rewards_RPPOwO_2))
    print(len(episode_rewards_RPPOwO_3))
    print(len(episode_rewards_RPPOwO_4))
    #===============================================================================================================

    print('loading completed')

    # episode_rewards_RPPOwO = (episode_rewards_RPPOwO_1 + episode_rewards_RPPOwO_2 + episode_rewards_RPPOwO_3 + episode_rewards_RPPOwO_4)
    episode_rewards_RPPOwO = (episode_rewards_RPPOwO_2 + episode_rewards_RPPOwO_4)
    # episode_rewards_RPPOwO = episode_rewards_RPPOwO_4
    #============================= Model Testing ===========================================

    # (1) Convergence Speed
    plt.figure(figsize=(10, 5))

    # Plot each series individually
    plt.plot(range(len(episode_rewards_RPPOwO)), episode_rewards_RPPOwO,
             label='PPO + LSTM')
    # plt.plot(range(len(episode_rewards_PPOwM)), episode_rewards_PPOwM,
    #          label='PPO + 2 Images')

    # Labeling the axes and the plot
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.title('Mean Rewards Across Episodes')
    plt.legend()
    plt.savefig("Convergence Speed.png")

    # (2) Test Results

    # Set parameters
    time_step_duration = 0.02  # Manually set the time step duration (commonly 0.02 seconds)
    success = 1
    episodes_test = 100

    #================================================================================
    # Evaluate Random Policy
    total_rewards = []
    total_times_random = []
    success_count_random = 0

    env = gym.make("Pendulum-v1")

    for _ in range(episodes_test):
        obs = env.reset()
        done = False
        episode_reward = 0
        total_time = 0.0

        while not done:
            # Take a random action, ensuring it is in the correct format
            action = np.array([env.action_space.sample()])
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            total_time += time_step_duration

        total_rewards.append(episode_reward)
        total_times_random.append(total_time)

        if total_time >= success:
            success_count_random += 1

    mean_reward_random = np.mean(total_rewards)
    std_reward_random = np.std(total_rewards)
    avg_time_random = np.mean(total_times_random)
    max_time_random = np.max(total_times_random)
    success_rate_random = success_count_random / episodes_test
    # ================================================================================

    # Define the function for testing
    def get_results(model, env, episodes_test=episodes_test, time_step_duration=time_step_duration, success=success):
        episode_rewards, episode_lengths = evaluate_policy(model, env, n_eval_episodes=episodes_test,
                                                           return_episode_rewards=True)

        # Calculate the mean reward and standard deviation of rewards
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # Calculate the mean and maximum episode length
        mean_episode_length = sum(episode_lengths) / len(episode_lengths)
        max_episode_length = max(episode_lengths)

        # Calculate the average and maximum time duration
        avg_time = mean_episode_length * time_step_duration
        max_time = max_episode_length * time_step_duration

        # Calculate the success rate
        # Calculate the success rate (episodes longer than 50 steps)
        success_count = sum([1 for x in episode_lengths if x * time_step_duration > success])
        success_rate = success_count / len(episode_lengths)

        return mean_reward, std_reward, avg_time, max_time, success_rate

    # Conduct the test for 2 models
    # Evaluate each model for custom metrics
    # Evaluate each model
    mean_reward_RPPOwO, std_reward_RPPOwO, avg_time_RPPOwO, max_time_RPPOwO, success_rate_RPPOwO = get_results(model_RPPOwO, env_si)
    # mean_reward_PPOwM, std_reward_PPOwM, avg_time_PPOwM, max_time_PPOwM, success_rate_PPOwM = get_results(model_PPOwM,env_mi)

    # Print results in tabular format
    print("\nTest Results:")
    print( f"{'Model':<10} {'Mean Reward':<20} {'Average Time (s)':<20} {'Max Time (s)':<15} {'Success Rate (' + str(success) + 's+)':<20}")
    print(f"{'Random':<10} {mean_reward_random:.2f} +- {std_reward_random:.2f} {'':<8} {avg_time_random:<20.2f} {max_time_random:<15.2f} {success_rate_random:<20.2%}")
    print( f"{'RPPOwO':<10} {mean_reward_RPPOwO:.2f} +- {std_reward_RPPOwO:.2f} {'':<8} {avg_time_RPPOwO:<20.2f} {max_time_RPPOwO:<15.2f} {success_rate_RPPOwO:<20.2%}")
    # print( f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8} {avg_time_PPOwM:<20.2f} {max_time_PPOwM:<15.2f} {success_rate_PPOwM:<20.2%}")

    # ============================= Model Testing ===========================================
    # Compare
    def count_parameters(model):
        # Access the underlying policy network's parameters
        return sum(p.numel() for p in model.policy.parameters() if
                   p.requires_grad)  # total number of trainable parameters in the neural network that the model uses for decision-making


    # Count parameters for RecurrentPPO
    total_params_RPPOwO_parameters = count_parameters(model_RPPOwO)
    # total_params_PPOwM_parameters = count_parameters(model_PPOwM)

    # # Print results in tabular format
    # print("\nTraining Results:")
    # print(f"{'Model':<10} {'Training Time (s)':<20} {'Total Parameters':<20}")
    # print(f"{'RPPOwO':<10} {training_time_RPPOwO:<20.2f} {total_params_RPPOwO_parameters:<20}")
    # # print(f"{'PPOwM':<10} {training_time_PPOwM:<20.2f} {total_params_PPOwM_parameters:<20}")

    ##################################################################################################
    # Save two tables
    # Define the output file name
    output_file = "test_and_training_results.txt"

    # Open the file in write mode
    with open(output_file, 'w') as file:
        # Write the test results table header
        file.write("Test Results:\n")
        file.write( f"{'Model':<10} {'Mean Reward':<20} {'Average Time (s)':<20} {'Max Time (s)':<15} {'Success Rate (' + str(success) + 's+)':<20}\n")

        # Write the test results table rows
        file.write( f"{'Random':<10} {mean_reward_random:.2f} +- {std_reward_random:.2f} {'':<8} {avg_time_random:<20.2f} {max_time_random:<15.2f} {success_rate_random:<20.2%}\n")
        file.write( f"{'RPPOwO':<10} {mean_reward_RPPOwO:.2f} +- {std_reward_RPPOwO:.2f} {'':<8} {avg_time_RPPOwO:<20.2f} {max_time_RPPOwO:<15.2f} {success_rate_RPPOwO:<20.2%}\n")
        # file.write( f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8} {avg_time_PPOwM:<20.2f} {max_time_PPOwM:<15.2f} {success_rate_PPOwM:<20.2%}\n")


    print(f"Test and Training results have been saved to {output_file}")


    # ====================================================================================")