import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

from gym_torcs import TorcsEnv

import matplotlib.pyplot as plt
from itertools import count
import utils as utils
import os

import cBCQ_model as cBCQ_model


def evaluate_policy(policy, eval_episodes=2):
    avg_reward = 0.

    avg_trackpos = 0.
    avg_speed =0.
    total_reward = []
    for _ in range(eval_episodes):
        score = 0
        ob, qq = env.reset()
        state = np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
        obs = state.reshape(1, state.shape[0])
        done = False

        episode_step =0
        while not done:
            episode_step=episode_step+1

            action = policy.select_action(obs)
            #obs, reward, done, _ = env.step(action)
            ob, reward, done, info, qq = env.step(action)
            score += reward
            state = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            obs = state.reshape(1, state.shape[0])

            avg_trackpos += ob.trackPos
            avg_speed += ob.speedX

            avg_reward += reward
            if done or episode_step>300:
                total_reward.append(score)
                print(total_reward)
                break
            # if episode_step>300:
            #     break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    print("total reward : ", total_reward)
    print("total reward mean: ", np.mean(total_reward))

    return avg_reward, avg_trackpos, avg_speed


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #file_name = "cBCQ_torcs_failed_75_percent"
    file_name = "cBCQ_torcs_normal_buffer"
    # buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    eval_mode = False
    max_timesteps = 5e4 #5e4 #1e5
    eval_freq = 3e3 #5e3

    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.random.seed(646) #1337 #131 222 646
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    state_dim = 29  # state의 생김새
    action_dim = 3  # action의 갯수
    max_action = env.action_space.high[0]  # 각 action의 최대값

    latent_dim=action_dim * 2

    # Initialize policy
    policy = cBCQ_model.BCQ(state_dim, action_dim, max_action,device)

    buffer_name = "./buffer_original_reward3/test_buffer"

    #load policy
    policy.load(f"./SaveModel/cBCQ_torcs_normal_buffer")

    evaluations = []
    trackpos_list = []
    speed_list = []

    episode_num = 0
    done = True

    # while training_iters < max_timesteps:
    #     print("evaluating,,,,")
    avg_reward, trackposes, speeds = evaluate_policy(policy, eval_episodes=20)


