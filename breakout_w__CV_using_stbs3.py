'''
Here, we solve Atari_Breakout by n-frames-stacked image input using PPO and RecurrnetPPO
'''

# Import Libraries
import gym
from gym import spaces

import cv2
#import pybulletgym
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

    env = gym.make("BreakoutNoFrameskip-v4")

    # Random policy
    total_rewards = []

    for _ in range(episodes_test):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Take a random action, ensuring it is in the correct format
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)  # Handle the additional truncated value
            episode_reward += reward
        total_rewards.append(episode_reward)

    mean_reward_random = np.mean(total_rewards)
    std_reward_random = np.std(total_rewards)

    return mean_reward_random, std_reward_random

#==================================================================================

# We only deal with one observation image per step

# Define the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),  # 84x84x4 -> 42x42x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 42x42x16 -> 21x21x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 21x21x32 -> 11x11x64
            nn.ReLU(),
            nn.Flatten(),  # 11x11x64 -> 7744
            nn.Linear(11 * 11 * 64, 1600),  # 7744 -> 1600
            nn.ReLU(),
            nn.Linear(1600, 128),  # 1600 -> 128
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


# Define the Environment Wrapper
class ImageObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env, autoencoder, width=84, height=84, frame_stack=4):
        super(ImageObservationWrapper, self).__init__(env)
        self.width = width
        self.height = height
        self.autoencoder = autoencoder
        self.frame_stack = frame_stack
        self.frames = []  # List to store last 4 frames
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)

    def preprocess(self, obs):

        # Convert to grayscale
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize the image
        img = cv2.resize(img, (self.width, self.height))
        return img

    def observation(self, obs):
        # Preprocess the current observation
        processed_img = self.preprocess(obs)

        # Add the new frame to the list
        self.frames.append(processed_img)

        # Remove the oldest frame (this keeps the list at 4 frames)
        self.frames.pop(0)

        # Stack frames along the channel dimension (4, 84, 84)
        img_stack = np.stack(self.frames, axis=0)

        # Convert numpy array to tensor
        img_stack = torch.tensor(img_stack, dtype=torch.float32).unsqueeze(0)  # (1, 4, 84, 84)

        # Pass through the autoencoder
        with torch.no_grad():
            img_encoded = self.autoencoder(img_stack).squeeze(0).numpy()  # (128,)

        return img_encoded

    def reset(self, **kwargs):
        # Reset the environment
        obs, info = self.env.reset(**kwargs)

        # Preprocess the first observation
        processed_img = self.preprocess(obs)

        # Stack 4 identical frames initially
        self.frames = [processed_img] * self.frame_stack

        # Encode the stacked frames
        img_obs = self.observation(obs)  # This call will use the stacked frames

        return img_obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        img_obs = self.observation(obs)
        return img_obs, reward, done, False, info

    def render(self, mode='human', **kwargs):
        return self.env.render(mode=mode, **kwargs)


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
    # type [python breakout_w__CV_using_stbs3.py] in your conda terminal to run this code

    # define the autoencoder
    autoencoder = Autoencoder()

    # define the environment
    env = gym.make("BreakoutNoFrameskip-v4")
    env = ImageObservationWrapper(env, autoencoder=autoencoder)
    env = DummyVecEnv([lambda: env])

    # set hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 20000
    episodes_test = 3


    #============================= [1] Recurrent PPO ===========================================

    # Define and Train the Model
    # Define the Model
    model_RPPOwO = RecurrentPPO('MlpLstmPolicy',  # not CnnLstmPolicy
                                env,
                                verbose=1,
                                device=device,
                                ent_coef=0.01,
                                use_sde=False,  # for discrete action space
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
    plt.savefig("RPPO_total_reward_per_episode_atari.png")

    #==================================== [2] PPO ===========================================
    # Define the Model
    model_PPOwO = PPO('MlpPolicy',  # Not CnnPolicy
                      env,
                      verbose=1,
                      device=device,
                      ent_coef=0.01,
                      use_sde=False,  # for discrete action space
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
    print('PPO Training End')

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
    plt.savefig("PPO_total_reward_per_episode_atari.png")


    #============================= [3] Model Testing ===========================================
    # Load the model and test them
    model_RPPOwO_loaded = PPO.load("rppo_model", env=env, device='cuda')
    model_PPOwO_loaded = PPO.load("ppo_model", env=env, device='cuda')

    # (1) Run the Random Policy Model
    #mean_reward_random, std_reward_random = random_policy()

    # (2) Test the RecurrentPPO Model
    mean_reward_RPPOwO, std_reward_RPPOwO = evaluate_policy(model_RPPOwO_loaded, env, n_eval_episodes=episodes_test)

    # (3) Test the PPO Model
    mean_reward_PPOwO, std_reward_PPOwO = evaluate_policy(model_PPOwO_loaded, env, n_eval_episodes=episodes_test)

    # print the results
    print("===============================================================================================")
    #print(f"Mean reward, random policy: {mean_reward_random} +/- {std_reward_random}")
    print(f"Mean reward, RecurrentPPO with one image per step: {mean_reward_RPPOwO} +/- {std_reward_RPPOwO}")
    print(f"Mean reward, PPO with one image per step: {mean_reward_PPOwO} +/- {std_reward_PPOwO}")
    print("===============================================================================================")