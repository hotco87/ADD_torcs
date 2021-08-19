import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
# from skimage.transform import resize
# from skimage.color import rgb2gray
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils as utils
import time
import yaml

from gym_torcs import TorcsEnv

import matplotlib.pyplot as plt
from itertools import count

from agent.ddpg import *


def main():  # True
    start_time = time.time()

    np.random.seed(1337)

    # env = TorcsEnv(vision=True, throttle=True, gear_change=False)
    env = TorcsEnv(vision=True, throttle=True, gear_change=False)
    print("Load parameters from env")
    obs_size = 29  # state의 생김새
    action_size = 3  # action의 갯수
    action_range = env.action_space.high[0]  # 각 action의 최대값

    print("obs size : {}, action_size : {} action_range : {}".format(
        obs_size, action_size, action_range))

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

    agent = DDPG(args_dict)

    agent.load_state_dict(torch.load('./SaveModel/3DDPG_save_original_reward399972.pth'))
    print("load")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    replay_buffer = utils.ReplayBuffer(state_dim=obs_size, action_dim=action_size, device=device)

    buffer_name = "test_buffer"

    recent_reward = deque(maxlen=100)
    frame = 0
    total_reward = []
    print("Torcs env start")
    total_step = 0
    # for e in count(1):  # range(args.episode):
    for e in range(20):  # range(args.episode):
        score = 0
        step = 0
        done = False
        if np.mod(e, 5) == 0:
            ob, rs = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob, rs = env.reset()

        state = np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))

        raw_state = np.hstack(
            (rs['angle'], rs['track'], rs['trackPos'], rs['speedY'], rs['speedX'],
             rs['speedZ'], rs['wheelSpinVel'], rs['rpm']))

        raw_state = raw_state.reshape(1, raw_state.shape[0])
        state = state.reshape(1, state.shape[0])

        while not done:
            total_step += 1
            step += 1
            frame += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            ob, reward, done, info, rs = env.step(action[0])
            # print(np.shape(action))
            # print(action[0]) # 3
            # print(reward)
            next_state = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            raw_next_state = np.hstack(
                (rs['angle'], rs['track'], rs['trackPos'], rs['speedY'], rs['speedX'],
                 rs['speedZ'], rs['wheelSpinVel'], rs['rpm']))

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action[0], reward, next_state, done)
            if step > 300:
                done = True

            # replay_bufferbuffer.add(state,action,next_state,reward,done,)
            replay_buffer.add(state, action, next_state, reward, done, raw_state, raw_next_state)

            score += reward
            raw_state = raw_next_state
            raw_state = raw_state.reshape(1, raw_state.shape[0])
            state = next_state
            state = state.reshape(1, state.shape[0])

            if frame > agent.batch_size:
                agent.train()
                agent.update_target_model()

            if done:
                total_reward.append(score)

                # if total_step % 1000:
                # #if e % 20 == 0:
                #     plt.plot(list(range(len(total_reward))), total_reward, c="b", lw=3, ls="--")
                #     plt.xlabel("Episode")
                #     plt.ylabel("Reward")
                #     plt.title("Reward Graph")
                #     fig = plt.gcf()
                #     # plt.show()
                #     fig.savefig('./SaveModel/reward_graph22.png')
                #     # plt.close()
                #     torch.save(agent.state_dict(), './SaveModel/3DDPG_save_original_reward3' + str(total_step) + '.pth')
                #     #replay_buffer.save(f"./buffer_original_reward3/{buffer_name}")
                #     print("buffer saved, total step :", total_step)

                recent_reward.append(score)

                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "   steps:", step,
                      "    recent reward:", np.mean(recent_reward))

            # if (total_step == 100000):
            #     replay_buffer.save(f"./buffer_original_reward3/{buffer_name}")
            #     print("buffer saved, total step :", total_step)
            #     end_time = time.time()
            #     total_time = start_time - end_time
            #     print("training ended")
            #     print("total time : ", total_time)
            #
            #     break
            if step > 300:
                print("step over")
                break

        if (total_step == 100000):
            break
            # if the mean of scores of last 10 episode is bigger than 400
            # stop training

    end_time = time.time()
    total_time = start_time - end_time
    print("training ended")
    print("total time : ", total_time)
    print("total reward : ", total_reward, np.mean(total_reward))


if __name__ == '__main__':
    main()
