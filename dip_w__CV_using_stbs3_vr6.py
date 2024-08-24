'''
vr3 -> vr4
*   Modified the parameters : ent_coef=0.01, use_sde=True, gamma=0.9, learning_rate=0.0001, batch_size=256
*   total_timesteps = 500000

vr4 -> vr5

*   Cropping size changed to 160
*   Pixel reduced to 36 x 36

vr5 -> vr6
*   Bring back the multiple-images environment
*   Compare 3 models : RecurrentPPO with multiple images, PPO with multiple images, and RecurrentPPO with one image
*   보고서 작성을 위한 결과 그래프 및 표 출력 (마지막 부분에)
'''

# Import Libraries
import gym
from gym import spaces
from collections import deque
from gym.spaces import Box
import time

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
# Here, we define the environment wrapper to modify the observation of it

# [1] Multiple Images Stacked Environment
class ImageFrameStack(gym.ObservationWrapper):
    def __init__(self, env, n_frames=4):
        super(ImageFrameStack, self).__init__(env)
        self.width = 36
        self.height = 36
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)  # keeps the last n_frames;
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, n_frames), dtype=np.uint8)

    # modify the reset method to build initial observation with multiple frames
    def reset(self):
        obs = self.env.reset()
        processed_obs = self._process_image(obs)

        # Fill the deque with the initial frame multiple times to start the episode with consistent frame stacks
        for _ in range(self.n_frames):
            self.frames.append(processed_obs)
        return self._get_observation()  # outputs the converted self.frames

    # processes the raw observation to an image vector, append it to the deque
    def observation(self, obs):
        processed_obs = self._process_image(obs)
        self.frames.append(processed_obs)  # Append the processed observation to the deque, automatically removing the oldest frame if full
        return self._get_observation()  # outputs the converted self.frames

    def _process_image(self, obs):
        # Capture the image from the environment
        img = self.env.render(mode='rgb_array')

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
        return img

    # converts four 2D arrays inside of deque into 3D array
    def _get_observation(self):
        # Stack the frames along the last axis

        # self.frames -> deque containing four 2D arrays of shape (36, 36)
        # np.stack(self.frames, axis=-1) -> 3D array of shape (36, 36, 4)
        return np.stack(self.frames, axis=-1)


# [2] Env with one image
class ImageInputWrapper(gym.ObservationWrapper):
    def __init__(self, env, width=36, height=36):
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
    # type [python dip_w__CV_using_stbs3_vr6.py] in your conda terminal to run this code


    # define the environment
    # observation with multiple images
    env_mi = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env_mi = ImageFrameStack(env_mi, n_frames=4)
    env_mi = DummyVecEnv([lambda: env_mi])

    # observation with single image
    env_si = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env_si = ImageInputWrapper(env_si)  # Creates an object of ImageInputWrapper class; Inputs the screen image
    env_si = DummyVecEnv([lambda: env_si])  # returns the env object; necessary in stablebaseline3

    # set hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = 500000


    #============================= [Model 1] Recurrent PPO with Multiple Images===========================

    # Define and Train the Model
    # Define the Model
    model_RPPOwM = RecurrentPPO('CnnLstmPolicy',
                                env_mi,
                                verbose=1,
                                device=device,
                                ent_coef=0.01,
                                use_sde=True,
                                gamma=0.9,
                                learning_rate=0.0001,
                                batch_size=256,
                                tensorboard_log="./logs/RPPOwM/")

    # Initialize the custom callback
    total_reward_logger_RPPOwM = TotalRewardLoggerCallback()

    # Train the model with the custom callback
    print('RecurrentPPO/M Training Start')
    start_time_RPPOwM = time.time()
    #######################################################################################
    model_RPPOwM.learn(total_timesteps=total_timesteps, callback=total_reward_logger_RPPOwM)
    #######################################################################################
    end_time_RPPOwM = time.time()
    print('RecurrentPPO/M Training End')
    training_time_RPPOwM = end_time_RPPOwM - start_time_RPPOwM


    #============================= [Model 2] PPO with Multiple Images ===========================
    # Define the Model
    model_PPOwM = PPO('CnnPolicy',
                      env_mi,
                      verbose=1,
                      device='cuda',
                      ent_coef=0.01,
                      use_sde=True,
                      gamma=0.9,
                      learning_rate=0.0001,
                      batch_size=256,
                      tensorboard_log="./logs/PPOwM/")
    '''
    This will automatically save logs to the specified directory (./logs/PPOwO/ in your case).
    To view these logs using TensorBoard: (1) Open a Terminal, 
    (2) Navigate to the directory where your logs are stored (in your case, ./logs/),
    (3) Execute the following command: tensorboard --logdir=./logs/ -> You'll get an url
    (4) Open TensorBoard in a Browser
    '''

    # Initialize the custom callback
    total_reward_logger_PPOwM = TotalRewardLoggerCallback()

    # Train the model with the custom callback
    print('PPO/M Training Start')
    start_time_PPOwM = time.time()
    #######################################################################################
    model_PPOwM.learn(total_timesteps=total_timesteps, callback=total_reward_logger_PPOwM)
    #######################################################################################
    end_time_PPOwM = time.time()
    print('PPO/M Training End')
    training_time_PPOwM = end_time_PPOwM - start_time_PPOwM

    # ============================= [Model 3] RecurrentPPO with One Image ===========================
    # Define the Model
    model_RPPOwO = RecurrentPPO('CnnLstmPolicy',
                                env_si,
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

    print('RPPO/O Training Start')
    start_time_RPPOwO = time.time()
    #######################################################################################
    model_RPPOwO.learn(total_timesteps=total_timesteps, callback=total_reward_logger_RPPOwO)
    #######################################################################################
    end_time_RPPOwO = time.time()
    print('RPPO/O Training End')
    training_time_RPPOwO = end_time_RPPOwO - start_time_RPPOwO

    #============================= Model Testing ===========================================

    # (1) Convergence Speed
    plt.figure(figsize=(10, 5))

    # Plot each series individually
    plt.plot(range(len(total_reward_logger_RPPOwM.episode_rewards)), total_reward_logger_RPPOwM.episode_rewards,
             label='RecurrentPPO with Multiple Images')
    plt.plot(range(len(total_reward_logger_PPOwM.episode_rewards)), total_reward_logger_PPOwM.episode_rewards,
             label='PPO with Multiple Image')
    plt.plot(range(len(total_reward_logger_RPPOwO.episode_rewards)), total_reward_logger_RPPOwO.episode_rewards,
             label='RecurrentPPO with One Imag')

    # Labeling the axes and the plot
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Comparison of Total Rewards Across Episodes for Different Training Models')
    plt.legend()
    plt.savefig("Convergence Speed.png")

    # (2) Test Results

    # Set parameters
    time_step_duration = 0.02  # Manually set the time step duration (commonly 0.02 seconds)
    success = 5
    episodes_test = 20

    #================================================================================
    # Evaluate Random Policy
    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    total_rewards = []
    total_times_random = []
    success_count_random = 0

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
    def evaluate_custom_metrics(model, env):
        total_times = []
        success_count = 0

        for _ in range(episodes_test):
            obs = env.reset()
            done = False
            total_time = 0.0

            while not done:
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, info = env.step(action)
                total_time += time_step_duration

            total_times.append(total_time)

            if total_time >= success:
                success_count += 1

        average_time = np.mean(total_times)
        max_time = np.max(total_times)
        success_rate = success_count / episodes_test

        return average_time, max_time, success_rate

    # Conduct the test for 3 models
    # Evaluate each model for custom metrics
    avg_time_RPPOwM, max_time_RPPOwM, success_rate_RPPOwM = evaluate_custom_metrics(model_RPPOwM, env_mi)
    avg_time_PPOwM, max_time_PPOwM, success_rate_PPOwM = evaluate_custom_metrics(model_PPOwM, env_mi)
    avg_time_RPPOwO, max_time_RPPOwO, success_rate_RPPOwO = evaluate_custom_metrics(model_RPPOwO, env_si)

    # Evaluate mean reward and standard deviation
    mean_reward_RPPOwM, std_reward_RPPOwM = evaluate_policy(model_RPPOwM, env_mi, n_eval_episodes=episodes_test,
                                                            return_episode_rewards=False)
    mean_reward_PPOwM, std_reward_PPOwM = evaluate_policy(model_PPOwM, env_mi, n_eval_episodes=episodes_test,
                                                          return_episode_rewards=False)
    mean_reward_RPPOwO, std_reward_RPPOwO = evaluate_policy(model_RPPOwO, env_si, n_eval_episodes=episodes_test,
                                                            return_episode_rewards=False)

    # Print results in tabular format
    print("\nTest Results:")
    print( f"{'Model':<10} {'Mean Reward':<20} {'Average Time (s)':<20} {'Max Time (s)':<15} {'Success Rate (' + str(success) + 's+)':<20}")
    print(f"{'Random':<10} {mean_reward_random:.2f} +- {std_reward_random:.2f} {'':<8} {avg_time_random:<20.2f} {max_time_random:<15.2f} {success_rate_random:<20.2%}")
    print( f"{'RPPOwM':<10} {mean_reward_RPPOwM:.2f} +- {std_reward_RPPOwM:.2f} {'':<8} {avg_time_RPPOwM:<20.2f} {max_time_RPPOwM:<15.2f} {success_rate_RPPOwM:<20.2%}")
    print( f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8} {avg_time_PPOwM:<20.2f} {max_time_PPOwM:<15.2f} {success_rate_PPOwM:<20.2%}")
    print( f"{'RPPOwO':<10} {mean_reward_RPPOwO:.2f} +- {std_reward_RPPOwO:.2f} {'':<8} {avg_time_RPPOwO:<20.2f} {max_time_RPPOwO:<15.2f} {success_rate_RPPOwO:<20.2%}")

    # ============================= Model Testing ===========================================
    # Compare
    def count_parameters(model):
        # Access the underlying policy network's parameters
        return sum(p.numel() for p in model.policy.parameters() if
                   p.requires_grad)  # total number of trainable parameters in the neural network that the model uses for decision-making


    # Count parameters for RecurrentPPO
    total_params_RPPOwM_parameters = count_parameters(model_RPPOwM)
    total_params_PPOwM_parameters = count_parameters(model_PPOwM)
    total_params_RPPOwO_parameters = count_parameters(model_RPPOwO)

    # Print results in tabular format
    print("\nTraining Results:")
    print(f"{'Model':<10} {'Training Time (s)':<20} {'Total Parameters':<20}")
    print(f"{'RPPOwM':<10} {training_time_RPPOwM:<20.2f} {total_params_RPPOwM_parameters:<20}")
    print(f"{'PPOwM':<10} {training_time_PPOwM:<20.2f} {total_params_PPOwM_parameters:<20}")
    print(f"{'RPPOwO':<10} {training_time_RPPOwO:<20.2f} {total_params_RPPOwO_parameters:<20}")

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
        file.write(  f"{'RPPOwM':<10} {mean_reward_RPPOwM:.2f} +- {std_reward_RPPOwM:.2f} {'':<8} {avg_time_RPPOwM:<20.2f} {max_time_RPPOwM:<15.2f} {success_rate_RPPOwM:<20.2%}\n")
        file.write( f"{'PPOwM  ':<10} {mean_reward_PPOwM:.2f} +- {std_reward_PPOwM:.2f} {'':<8} {avg_time_PPOwM:<20.2f} {max_time_PPOwM:<15.2f} {success_rate_PPOwM:<20.2%}\n")
        file.write( f"{'RPPOwO':<10} {mean_reward_RPPOwO:.2f} +- {std_reward_RPPOwO:.2f} {'':<8} {avg_time_RPPOwO:<20.2f} {max_time_RPPOwO:<15.2f} {success_rate_RPPOwO:<20.2%}\n")

        # Write a separator between sections
        file.write("\n")

        # Write the training results table header
        file.write("Training Results:\n")
        file.write(f"{'Model':<10} {'Training Time (s)':<20} {'Total Parameters':<20}\n")

        # Write the training results table rows
        file.write(f"{'RPPOwM':<10} {training_time_RPPOwM:<20.2f} {total_params_RPPOwM_parameters:<20}\n")
        file.write(f"{'PPOwM ':<10} {training_time_PPOwM:<20.2f} {total_params_PPOwM_parameters:<20}\n")
        file.write(f"{'RPPOwO':<10} {training_time_RPPOwO:<20.2f} {total_params_RPPOwO_parameters:<20}\n")

    print(f"Test and Training results have been saved to {output_file}")


    # ====================================================================================")