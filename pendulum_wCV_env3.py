'''
This script trains and tests both the RecurrentPPO and PPO models for the Image Input Environment.
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
    # type [python pendulum_wCV_env3.py] in your conda terminal to run this code


    # define the environment

    # observation with single image
    env_si = gym.make('Pendulum-v1')
    env_si = ImageObservationWrapper(env_si)  # Creates an object of ImageInputWrapper class; Inputs the screen image
    env_si = DummyVecEnv([lambda: env_si])  # returns the env object; necessary in stablebaseline3

    # observation with multiple images
    env_mi = VecFrameStack(env_si, n_stack=4)

    # set hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 1000000


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
                                clip_range=0.2,
                                tensorboard_log="./logs/RPPOwO/")

    # Initialize the custom callback
    mean_reward_logger_RPPOwO = MeanRewardLoggerCallback()

    # Train the model with the custom callback
    print('RecurrentPPO/O Training Start')
    start_time_RPPOwO = time.time()
    #######################################################################################
    model_RPPOwO.learn(total_timesteps=total_timesteps, callback=mean_reward_logger_RPPOwO)
    #######################################################################################
    end_time_RPPOwO = time.time()
    print('RecurrentPPO/O Training End')
    training_time_RPPOwO = end_time_RPPOwO - start_time_RPPOwO

    # Save the trained model
    model_RPPOwO.save("RPPOwO_model")

    with open("episode_rewards_RPPOwO.pkl", "wb") as file:  # check the path before run!
        pickle.dump(mean_reward_logger_RPPOwO.episode_rewards, file)

    with open("training_time_RPPOwO.pkl", "wb") as file:  # check the path before run!
        pickle.dump(training_time_RPPOwO, file)

    # Save the total reward per episode figure - for checking purpose
    plt.figure(figsize=(10, 5))
    plt.plot(mean_reward_logger_RPPOwO.episode_rewards, label='Mean Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('<RecurrentPPO with one image>\n Mean Reward per Episode During Training')
    plt.legend()
    plt.savefig("RPPO_mean_reward_per_episode.png")

    #============================= [Model 2] PPO with Multiple Images ===========================
    # Define the Model
    model_PPOwM = PPO('CnnPolicy',
                      env_mi,  # env with multiple images
                      verbose=1,
                      device=device,
                      ent_coef=0.01,  # Entropy coefficient
                      use_sde=True,
                      gamma=0.9,
                      gae_lambda=0.95,
                      learning_rate=0.0001,
                      clip_range=0.2,
                      batch_size=256,
                      tensorboard_log="./logs/PPOwM/")

    # Initialize the custom callback
    mean_reward_logger_PPOwM = MeanRewardLoggerCallback()

    # Train the model with the custom callback
    print('PPO/M Training Start')
    start_time_PPOwM = time.time()
    #######################################################################################
    model_PPOwM.learn(total_timesteps=total_timesteps, callback=mean_reward_logger_PPOwM)
    #######################################################################################
    end_time_PPOwM = time.time()
    print('PPO/M Training End')
    training_time_PPOwM = end_time_PPOwM - start_time_PPOwM

    # Save the trained model, total reward per episode, and training time
    model_PPOwM.save("PPOwM_model")

    with open("episode_rewards_PPOwM.pkl", "wb") as file:  # check the path before run!
        pickle.dump(mean_reward_logger_PPOwM.episode_rewards, file)

    with open("training_time_PPOwM.pkl", "wb") as file:  # check the path before run!
        pickle.dump(training_time_PPOwM, file)

    # Save the total reward per episode figure - for checking purpose
    plt.figure(figsize=(10, 5))
    plt.plot(mean_reward_logger_PPOwM.episode_rewards, label='Mean Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('<PPO with multiple images>\n Mean Reward per Episode During Training')
    plt.legend()
    plt.savefig("PPO_mean_reward_per_episode.png")

    #============================= Model Testing ===========================================

    # (1) Convergence Speed
    plt.figure(figsize=(10, 5))

    # Plot each series individually
    plt.plot(range(len(mean_reward_logger_RPPOwO.episode_rewards)), mean_reward_logger_RPPOwO.episode_rewards,
             label='PPO + LSTM')
    plt.plot(range(len(mean_reward_logger_PPOwM.episode_rewards)), mean_reward_logger_PPOwM.episode_rewards,
             label='PPO with 2 Images')


    # Labeling the axes and the plot
    plt.xlabel('Episodes')
    plt.ylabel('Mean Return')
    plt.title('Pendulum-v1')
    plt.legend()
    plt.savefig("Convergence Speed.png")

    # (2) Test Results

    # Set parameters
    time_step_duration = 0.02  # Manually set the time step duration (commonly 0.02 seconds)
    success = 1
    episodes_test = 100

    #================================================================================
    # Evaluate Random Policy
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
    mean_reward_PPOwM, std_reward_PPOwM, avg_time_PPOwM, max_time_PPOwM, success_rate_PPOwM = get_results(model_PPOwM,env_mi)

    # Print results in tabular format
    print("\nTest Results:")
    print( f"{'Model':<10} {'Mean Reward':<20} {'Average Time (s)':<20} {'Max Time (s)':<15} {'Success Rate (' + str(success) + 's+)':<20}")
    print(f"{'Random':<10} {mean_reward_random:.2f} +- {std_reward_random:.2f} {'':<8} {avg_time_random:<20.2f} {max_time_random:<15.2f} {success_rate_random:<20.2%}")
    print( f"{'RPPOwO':<10} {mean_reward_RPPOwO:.2f} +- {std_reward_RPPOwO:.2f} {'':<8} {avg_time_RPPOwO:<20.2f} {max_time_RPPOwO:<15.2f} {success_rate_RPPOwO:<20.2%}")
    print( f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8} {avg_time_PPOwM:<20.2f} {max_time_PPOwM:<15.2f} {success_rate_PPOwM:<20.2%}")

    # ============================= Model Testing ===========================================
    # Compare
    def count_parameters(model):
        # Access the underlying policy network's parameters
        return sum(p.numel() for p in model.policy.parameters() if
                   p.requires_grad)  # total number of trainable parameters in the neural network that the model uses for decision-making


    # Count parameters for RecurrentPPO
    total_params_RPPOwO_parameters = count_parameters(model_RPPOwO)
    total_params_PPOwM_parameters = count_parameters(model_PPOwM)

    # Print results in tabular format
    print("\nTraining Results:")
    print(f"{'Model':<10} {'Training Time (s)':<20} {'Total Parameters':<20}")
    print(f"{'RPPOwO':<10} {training_time_RPPOwO:<20.2f} {total_params_RPPOwO_parameters:<20}")
    print(f"{'PPOwM':<10} {training_time_PPOwM:<20.2f} {total_params_PPOwM_parameters:<20}")

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
        file.write( f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8} {avg_time_PPOwM:<20.2f} {max_time_PPOwM:<15.2f} {success_rate_PPOwM:<20.2%}\n")

        # Write a separator between sections
        file.write("\n")

        # Write the training results table header
        file.write("Training Results:\n")
        file.write(f"{'Model':<10} {'Training Time (s)':<20} {'Total Parameters':<20}\n")

        # Write the training results table rows
        file.write(f"{'RPPOwO':<10} {training_time_RPPOwO:<20.2f} {total_params_RPPOwO_parameters:<20}\n")
        file.write(f"{'PPOwM ':<10} {training_time_PPOwM:<20.2f} {total_params_PPOwM_parameters:<20}\n")

    print(f"Test and Training results have been saved to {output_file}")


    # ====================================================================================")