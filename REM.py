import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym_torcs import TorcsEnv

import matplotlib.pyplot as plt
from itertools import count
import utils as utils
import os

import REM_model as  REM_model


def evaluate_policy(policy, eval_episodes=3):
    avg_reward = 0.
    for _ in range(eval_episodes):
        ob = env.reset()
        state = np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
        obs = state.reshape(1, state.shape[0])
        done = False

        episode_step =0
        while not done or episode_step <400:
            episode_step=episode_step+1

            action = policy.select_action(obs)
            #obs, reward, done, _ = env.step(action)
            ob, reward, done, info = env.step(action)
            state = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            obs = state.reshape(1, state.shape[0])

            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_name = "REM_torcs"
    # buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    max_timesteps =5e6
    eval_freq = 4e3

    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.random.seed(1337)
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    state_dim = 29  # state의 생김새
    action_dim = 3  # action의 갯수
    max_action = env.action_space.high[0]  # 각 action의 최대값

    # Initialize policy
    policy = REM_model.REM(state_dim, action_dim, max_action)

    buffer_name = "./buffers_trained/test_buffer"
    # Load buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load(buffer_name)

    # policy.load_state_dict(torch.load('./SaveModel/3DDPG_save.pth'))

    evaluations = []

    episode_num = 0
    done = True

    training_iters = 0
    while training_iters < max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(eval_freq))

        print("evaluating,,,,")
        evaluations.append(evaluate_policy(policy))
        np.save(results_dir + "/" + file_name, evaluations)

        training_iters += eval_freq
        print("Training iterations: " + str(training_iters))




