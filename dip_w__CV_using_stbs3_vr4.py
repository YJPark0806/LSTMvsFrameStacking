'''
vr3 -> vr4
*   Modified the parameters : ent_coef=0.01, use_sde=True, gamma=0.9, learning_rate=0.0001, batch_size=256
*   total_timesteps = 500000
'''

# Import Libraries
import gym
from gym import spaces

import cv2
import pybulletgym
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
#from stable_baselines3.common.envs import DummyVecEnv, VecTransposeImage

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim import Adam
from torch.distributions import MultivariateNormal



#==================================================================================
# First, we calculate the Mean reward of the Random Policy to set the baseline.
def random_policy():

    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')

    # Random policy
    total_rewards = []

    for _ in range(episodes_test):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Take a random action, ensuring it is in the correct format
            action = np.array([env.action_space.sample()])
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        total_rewards.append(episode_reward)

    mean_reward_random = np.mean(total_rewards)
    std_reward_random = np.std(total_rewards)

    return mean_reward_random, std_reward_random

#==================================================================================

# We only deal with one observation image per step

# Define the Environment Wrapper
class ImageInputWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=64, height=64):
        super(ImageInputWrapper, self).__init__(env)  # execute the parent class's init method
        self.env = env
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(height, width, 1), dtype=np.uint8)  # modify the observation space

    def observation(self, obs):
        img = self.env.render(mode='rgb_array')  # captures an image of the environment

        # Crop the image to focus on the pendulum
        # assuming the pendulum is centered in the middle
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        crop_size = 160
        img = img[center_y - crop_size//2:center_y + crop_size//2, center_x - crop_size//2:center_x + crop_size//2]

        # # Plot the image
        #   # NOTE THAT When you display a grayscale image using imshow,
        #   # Matplotlib uses a colormap to map the single-channel grayscale values to colors
        # plt.imshow(img)
        # plt.axis('off')  # Turn off the axis labels
        # plt.show(block=False)  # Non-blocking show
        # plt.pause(0.001)  # Pause to allow the plot to updat

        # Process the image (resize, grayscale, normalize)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.width, self.height))
        # Add a channel dimension to the image (from (height, width) to (height, width, 1)), to make it compatible with CnnPolciy
        img = img[:, :, None]


        return img


# Custom callback class to track the cumulative reward per episode
class TotalRewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TotalRewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Accumulate rewards using self.locals['rewards'] directly
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward

        # Check if the episode is done
        if self.locals['dones'][0]:
            # Log the total reward for this episode
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

        return True


#================================================================================================

if __name__ == '__main__':
    # type [python dip_w__CV_using_stbs3_vr4.py] in your conda terminal to run this code


    # define the environment
    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env = ImageInputWrapper(env)  # Creates an object of ImageInputWrapper class; Inputs the screen image
    env = DummyVecEnv([lambda: env])  # returns the env object; necessary in stablebaseline3

    # set hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 500000
    episodes_test = 50


    #============================= [1] Recurrent PPO ===========================================

    # Define and Train the Model
    # Define the Model
    model_RPPOwO = RecurrentPPO('CnnLstmPolicy',
                                env,
                                verbose=1,
                                device=device,
                                ent_coef=0.01,
                                use_sde=True,
                                gamma=0.9,
                                learning_rate=0.0001,
                                batch_size=256,
                                tensorboard_log="./logs/RPPOwO/")

    # Initialize the custom callback
    total_reward_logger_RPPOwO = TotalRewardLoggerCallback()

    # Train the model with the custom callback
    print('RecurrentPPO Training Start')
    model_RPPOwO.learn(total_timesteps=total_timesteps, callback=total_reward_logger_RPPOwO)
    print('RecurrentPPO Training End')

    # Save the trained model
    model_RPPOwO.save("rppo_model")

    # Save the total reward per episode figure
    plt.figure(figsize=(10, 5))
    plt.plot(total_reward_logger_RPPOwO.episode_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('<RecurrentPPO with one image>\n Total Reward per Episode During Training')
    plt.legend()
    plt.savefig("RPPO_total_reward_per_episode.png")

    #==================================== [2] PPO ===========================================
    # Define the Model
    model_PPOwO = PPO('CnnPolicy',
                      env,
                      verbose=1,
                      device=device,
                      ent_coef=0.01,
                      use_sde=True,
                      gamma=0.9,
                      learning_rate=0.0001,
                      batch_size=256,
                      tensorboard_log="./logs/PPOwO/")
    '''
    This will automatically save logs to the specified directory (./logs/PPOwO/ in your case).
    To view these logs using TensorBoard: (1) Open a Terminal, 
    (2) Navigate to the directory where your logs are stored (in your case, ./logs/),
    (3) Execute the following command: tensorboard --logdir=./logs/ -> You'll get an url
    (4) Open TensorBoard in a Browser
    '''

    # Initialize the custom callback
    total_reward_logger_PPOwO = TotalRewardLoggerCallback()

    # Train the model with the custom callback
    print('PPO Training Start')
    model_PPOwO.learn(total_timesteps=total_timesteps, callback=total_reward_logger_PPOwO)
    print('PPO Training Start')

    # Save the trained model
    model_PPOwO.save("ppo_model")
    '''
    This will save the model in a file called ppo_model.zip in the current working directory.
    You can later load the model using: model_PPOwO = PPO.load("ppo_model")
    '''

    # Save the total reward per episode figure
    plt.figure(figsize=(10, 5))
    plt.plot(total_reward_logger_PPOwO.episode_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('<PPO with one image>\n Total Reward per Episode During Training')
    plt.legend()
    plt.savefig("PPO_total_reward_per_episode.png")


    #============================= [3] Model Testing ===========================================
    # Load the model and test them
    model_RPPOwO_loaded = PPO.load("rppo_model", env=env, device='cuda')
    model_PPOwO_loaded = PPO.load("ppo_model", env=env, device='cuda')

    # (1) Run the Random Policy Model
    mean_reward_random, std_reward_random = random_policy()

    # (2) Test the RecurrentPPO Model
    mean_reward_RPPOwO, std_reward_RPPOwO = evaluate_policy(model_RPPOwO_loaded, env, n_eval_episodes=episodes_test)

    # (3) Test the PPO Model
    mean_reward_PPOwO, std_reward_PPOwO = evaluate_policy(model_PPOwO_loaded, env, n_eval_episodes=episodes_test)

    # print the results
    print("===============================================================================================")
    print(f"Mean reward, random policy: {mean_reward_random} +/- {std_reward_random}")
    print(f"Mean reward, RecurrentPPO with one image per step: {mean_reward_RPPOwO} +/- {std_reward_RPPOwO}")
    print(f"Mean reward, PPO with one image per step: {mean_reward_PPOwO} +/- {std_reward_PPOwO}")
    print("===============================================================================================")