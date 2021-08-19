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

import cBCQ_augmentation_model as cBCQ_model


def evaluate_policy(policy, eval_episodes=2):
    avg_reward = 0.

    avg_trackpos = 0.
    avg_speed =0.

    for _ in range(eval_episodes):
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
            state = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            obs = state.reshape(1, state.shape[0])

            avg_trackpos += ob.trackPos
            avg_speed += ob.speedX

            avg_reward += reward
            if episode_step>300:
                break

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_trackpos, avg_speed


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #file_name = "cBCQ_torcs_failed_75_percent"
    file_name = "cBCQ_torcs_vae_buffer"
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
    buffer_name2 = "./buffer_original_reward3/vae_buffer"
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
    replay_buffer.load2(buffer_name,buffer_name2)

    #load policy
    #policy.load(f"./SaveModel/cBCQ_")

    evaluations = []
    trackpos_list = []
    speed_list = []

    episode_num = 0
    done = True

    training_iters = 0
    eval_iters = 0
    while training_iters < max_timesteps:
        pol_vals = policy.train(replay_buffer, iterations=int(eval_freq))

        print("evaluating,,,,")
        avg_reward, trackposes, speeds = evaluate_policy(policy)
        trackpos_list.append(trackposes)
        speed_list.append(speeds)
        evaluations.append(avg_reward)
        np.save(results_dir + "/" + file_name, evaluations)

        # np.save(results_dir + "/" + file_name + "_tracpos", trackpos_list)
        # np.save(results_dir + "/" + file_name + "_speeds", speed_list)

        training_iters += eval_freq
        print("Training iterations: " + str(training_iters))

        policy.save(f"./SaveModel/"+file_name)
        print("Saved Model")

        end_time = time.time()
        total_time = end_time - start_time
        print("training ended")
        print("total time :", total_time)

        if training_iters > 50000:
            policy.save(f"./SaveModel/"+file_name)
            print("Saved Model")

            break
    end_time = time.time()
    total_time = end_time - start_time
    print("training ended")
    print("total time :", total_time)

        # if training_iters%100000 == 0:
        #     plt.plot(list(range(len(evaluations))), evaluations, c="b", lw=3, ls="--")
        #     plt.show()

    # while eval_mode:
    #     print("eval mode on")
    #     avg_reward, trackposes, speeds = evaluate_policy(policy, eval_episodes=1)
    #     trackpos_list.append(trackposes)
    #     speed_list.append(speeds)
    #     evaluations.append(avg_reward)
    #
    #     np.save(results_dir + "/" + file_name, evaluations)
    #     np.save(results_dir + "/" + file_name + "_tracpos", trackpos_list)
    #     np.save(results_dir + "/" + file_name + "_speeds", speed_list)
    #
    #     eval_iters+=1
    #


    #     if eval_iters == 100:


    #         break



