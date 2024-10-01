'''
This script is designed to train RecurrentPPO for the Image Input Environment.

To handle potential interruptions during training with a large total timestep,
the code includes four savepoints to store the model and total reward at intervals.

After training is complete, testing can be performed using the 'pendulum_wCV_RPPO_testing' file.
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
# Note : For env with multiple images input, we wrap this again with VecFrameStack
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


# Custom callback class to track the cumulative reward per episode
class MeanRewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MeanRewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.mean_rewards = []
        self.episode_reward = 0.0
        self.episode_count = 0

    def _on_step(self) -> bool:
        self.episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.episode_reward)
            self.episode_count += 1
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.mean_rewards.append(mean_reward)
            self.episode_reward = 0.0
        return True


#================================================================================================

if __name__ == '__main__':
    # type [python pendulum_wCV_RPPO_training.py] in your conda terminal to run this code


    # define the environment

    # observation with single image
    env_si = gym.make('Pendulum-v1')
    env_si = ImageObservationWrapper(env_si)  # Creates an object of ImageInputWrapper class; Inputs the screen image
    env_si = DummyVecEnv([lambda: env_si])  # returns the env object; necessary in stablebaseline3

    # # observation with multiple images
    # env_mi = VecFrameStack(env_si, n_stack=2)  # n_stack is 2

    # set hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 10000
    milestone = total_timesteps // 4


    #============================= [Model 1] Recurrent PPO with Multiple Images===========================

    # Define and Train the Model
    # Define the Model
    model_RPPOwO = RecurrentPPO('CnnLstmPolicy',
                                env_si,  # env with single image input
                                verbose=1,
                                device=device,
                                ent_coef=0.01,
                                use_sde=True,
                                gamma=0.88,
                                gae_lambda=0.97,
                                learning_rate=0.00005,
                                batch_size=256,
                                clip_range=0.2)

    # Initialize the custom callback
    mean_reward_logger_RPPOwO = MeanRewardLoggerCallback()

    # Off-comment This part if you are starting from the middle
    #===========================================================================================================
    # # Load the model from the checkpoint
    # model_RPPOwO = RecurrentPPO.load(f"./models/RPPOwO_checkpoint_5000.zip", env=env_si, device=device)
    #
    # # Load the mean reward logger's episode rewards
    # with open(f"./logs/RPPOwO_mean_rewards_5000.pkl", "rb") as f:
    #     mean_reward_logger_RPPOwO.episode_rewards = pickle.load(f)
    # ===========================================================================================================

    # Train the model in 4 parts, saving the model and the logger at each milestone
    '''
    If I have none ->  range(4)
    If I have ~2500 -> range(1,4)
    If I have ~5000 -> range(2,4)
    If I have ~7500 -> range(3,4)
    '''
    for i in range(4):
        print(f'RecurrentPPO/O Training Part {i + 1} Start')

        # Calculate the current milestone for this part
        current_timesteps = milestone * (i + 1)

        # Train the model for the current milestone
        model_RPPOwO.learn(total_timesteps=milestone, callback=mean_reward_logger_RPPOwO, reset_num_timesteps=False)

        # Save the model at the current milestone
        model_RPPOwO.save(f"./models/RPPOwO_checkpoint_{current_timesteps}")

        # Save the mean reward logger's episode rewards
        with open(f"./logs/RPPOwO_mean_rewards_{current_timesteps}.pkl", "wb") as f:
            pickle.dump(mean_reward_logger_RPPOwO.mean_rewards, f)

        print(f'RecurrentPPO/O Training Part {i + 1} End - Model and rewards saved at {current_timesteps} timesteps')

    # Save the total reward per episode figure - for checking purpose
    plt.figure(figsize=(10, 5))
    plt.plot(mean_reward_logger_RPPOwO.mean_rewards, label='Mean Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('<RecurrentPPO with one image>\n Mean Reward per Episode During Training')
    plt.legend()
    plt.savefig("RPPO_mean_reward_per_episode.png")

    # #============================= [Model 2] PPO with Multiple Images ===========================
    # # Define the Model
    # model_PPOwM = PPO('CnnPolicy',
    #                   env_mi,  # env with multiple images
    #                   verbose=1,
    #                   device=device,
    #                   ent_coef=0.02,  # Entropy coefficient
    #                   use_sde=True,
    #                   gamma=0.9,
    #                   gae_lambda=0.97,
    #                   learning_rate=0.00005,
    #                   clip_range=0.1,
    #                   batch_size=256,
    #                   tensorboard_log="./logs/PPOwM/")
    #
    # # Initialize the custom callback
    # mean_reward_logger_PPOwM = MeanRewardLoggerCallback()
    #
    # # Train the model with the custom callback
    # print('PPO/M Training Start')
    # start_time_PPOwM = time.time()
    # #######################################################################################
    # model_PPOwM.learn(total_timesteps=total_timesteps, callback=mean_reward_logger_PPOwM)
    # #######################################################################################
    # end_time_PPOwM = time.time()
    # print('PPO/M Training End')
    # training_time_PPOwM = end_time_PPOwM - start_time_PPOwM
    #
    # # Save the trained model, total reward per episode, and training time
    # model_PPOwM.save("PPOwM_model")
    #
    # with open("episode_rewards_PPOwM.pkl", "wb") as file:  # check the path before run!
    #     pickle.dump(mean_reward_logger_PPOwM.episode_rewards, file)
    #
    # with open("training_time_PPOwM.pkl", "wb") as file:  # check the path before run!
    #     pickle.dump(training_time_PPOwM, file)
    #
    # # Save the total reward per episode figure - for checking purpose
    # plt.figure(figsize=(10, 5))
    # plt.plot(mean_reward_logger_PPOwM.episode_rewards, label='Mean Reward per Episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Mean Reward')
    # plt.title('<PPO with multiple images>\n Mean Reward per Episode During Training')
    # plt.legend()
    # plt.savefig("PPO_mean_reward_per_episode.png")



    # ====================================================================================")