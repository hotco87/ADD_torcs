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

import ddpg_collect_data as ddpg_model
import cBCQ_model as cBCQ_model
import sys
import gym
import torch
import pylab
import random
from collections import deque
from datetime import datetime
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray

from gym_torcs import TorcsEnv

from itertools import count

def evaluate_policy(policy, eval_episodes=1):
    avg_reward = 0.

    avg_trackpos = 0.
    avg_speed =0.
    avg_theta =0.

    s_list = []
    t_list = []
    theta_list = []
    var_s = 0.
    var_t = 0.
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
            #action = policy.get_action(obs)

            ob, reward, done, info, qq = env.step(action)
            #ob, reward, done, info, qq = env.step(action[0])
            state = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            obs = state.reshape(1, state.shape[0])

            s_list.append(ob.speedX)
            t_list.append(np.abs(ob.trackPos))
            theta_list.append(ob.angle)

            avg_trackpos += np.abs(ob.trackPos)
            avg_speed += ob.speedX
            avg_theta += ob.angle

            avg_reward += reward
            if episode_step>300 or done:
                avg_trackpos =avg_trackpos/episode_step
                avg_speed = avg_speed/episode_step
                break

    avg_reward /= eval_episodes
    var_s =np.var(s_list)
    var_t =np.var(t_list)
    var_theta = np.var(theta_list)

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_trackpos, avg_speed, var_s,var_t , avg_theta, var_theta

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_name = "test"
    # buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: " + file_name)
    print("---------------------------------------")

    eval_mode = False
    max_timesteps =30
    eval_freq = 1

    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    np.random.seed(0) #1337 #423 #0 #8423 #54
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    state_dim = 29  # state의 생김새
    action_dim = 3  # action의 갯수


    max_action = env.action_space.high[0]  # 각 action의 최대값

    latent_dim=action_dim * 2

    obs_size = 29  # state의 생김새
    action_size = 3  # action의 갯수
    action_range = env.action_space.high[0]  # 각 action의 최대값
    args_dict = {}
    args_dict['memory_size'] = 500000
    args_dict['batch_size'] = 64
    args_dict['actor_lr'] = 1e-4
    args_dict['critic_lr'] = 1e-3
    args_dict['gamma'] = 0.99
    args_dict['decay'] = 1e-2
    args_dict['tau'] = 0.001
    args_dict['action_size'] = action_size
    args_dict['obs_size'] = obs_size
    args_dict['action_range'] = action_range

    # Initialize policy

    policy = cBCQ_model.BCQ(obs_size, action_size, action_range,device)
    #policy = ddpg_model.DDPG(args_dict)

    #load_policy
    policy.load(f"./SaveModel/cBCQ_failed_10_percent")
    #policy.load_state_dict(torch.load('./SaveModel/3DDPG_save_original_reward2.pth'))

    evaluations = []
    trackpos_list = []
    speed_list = []
    theta_list = []

    episode_num = 0

    eval_iters = 0

    var_s_list=[]
    var_t_list=[]
    var_theta_list =[]
    var_s =0.
    var_t =0.
    var_theta =0.

    print("eval mode on")
    while True:
        avg_reward, trackpos, speed,var_s,var_t, theta, var_theta = evaluate_policy(policy, eval_episodes=1)
        trackpos_list.append(trackpos)


        speed_list.append(speed)
        evaluations.append(avg_reward)
        var_s_list.append(var_s)
        var_t_list.append(var_t)
        var_theta_list.append(var_theta)
        theta_list.append(theta)

        np.save(results_dir + "/" + file_name, evaluations)
        np.save(results_dir + "/" + file_name + "_tracpos", trackpos_list)
        np.save(results_dir + "/" + file_name + "_speeds", speed_list)
        np.save(results_dir + "/" + file_name + "_var_s", var_s_list)
        np.save(results_dir + "/" + file_name + "_var_t", var_t_list)
        #np.save(results_dir + "/" + file_name + "_theta", theta_list)
        #np.save(results_dir + "/" + file_name + "_var_theta", var_theta_list)

        eval_iters+=1
        print("eval iters : ",eval_iters)

        if eval_iters == 30:
            print("eval ended")
            break





