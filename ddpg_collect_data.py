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

from gym_torcs import TorcsEnv

import matplotlib.pyplot as plt
from itertools import count


class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.6, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


HIDDEN1 = 300
HIDDEN2 = 600


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, action_range):
        self.action_range = action_range
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
        )
        self.steering = nn.Sequential(
            nn.Linear(HIDDEN2, 1),
            nn.Tanh()
        )
        self.accel = nn.Sequential(
            nn.Linear(HIDDEN2, 1),
            nn.Sigmoid()
        )
        self.breaks = nn.Sequential(
            nn.Linear(HIDDEN2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, action_size):
        x = self.network(x)
        if action_size == 3:
            actions = torch.cat([self.steering(x), self.accel(x), self.breaks(x)], dim=1)
        elif action_size == 2:
            actions = torch.cat([self.steering(x), self.accel(x)], dim=1)
        else:
            actions = torch.cat([self.steering(x)], dim=1)

        return actions


# tensorflow .dense = fully connected
# 예제
# Dense(100, activation='relu')([2,1])
# 입력이 모양이 [2,1]이고 출력이 100개인 fully connected layer
# 출력값을 주기 전 relu를 통과함
class Critic(nn.Module):
    def __init__(self, obs_size, action_size, action_range):
        self.action_range = action_range
        super(Critic, self).__init__()
        self.before_action = nn.Sequential(
            nn.Linear(obs_size, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2)
        )
        self.fc1 = nn.Linear(action_size, HIDDEN2)
        self.after_action = nn.Sequential(
            nn.Linear(HIDDEN2 + HIDDEN2, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, action_size)
        )

    def forward(self, x, actions):
        x = self.before_action(x)
        actions = self.fc1(actions)
        x = torch.cat([x, actions], dim=1)
        x = self.after_action(x)
        return x


class DDPG(nn.Module):
    def __init__(self, options):
        super(DDPG, self).__init__()
        # hyperparameter
        self.memory_size = options.get('memory_size', 10000000)
        self.action_size = options.get('action_size')
        self.action_range = options.get('action_range')
        self.obs_size = options.get('obs_size')
        self.batch_size = options.get('batch_size')

        self.actor_lr = options.get('actor_lr')
        self.critic_lr = options.get('critic_lr')
        self.gamma = options.get('gamma')
        self.decay = options.get('decay')
        self.tau = options.get('tau')

        # actor model
        self.actor = Actor(self.obs_size, self.action_size, self.action_range)
        self.actor_target = Actor(self.obs_size, self.action_size, self.action_range)

        # critic model
        self.critic = Critic(self.obs_size, self.action_size, self.action_range)
        self.critic_target = Critic(self.obs_size, self.action_size, self.action_range)

        # memory(uniformly)
        self.memory = deque(maxlen=self.memory_size)

        # explortion
        self.ou = OrnsteinUhlenbeckActionNoise(theta=0.60, sigma=0.30,
                                               mu=0.0, action_dim=self.action_size)
        self.ou2 = OrnsteinUhlenbeckActionNoise(theta=1.0, sigma=0.10,
                                                mu=0.0, action_dim=self.action_size)
        self.ou3 = OrnsteinUhlenbeckActionNoise(theta=1.0, sigma=0.50,
                                                mu=-0.1, action_dim=self.action_size)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # initialize target model
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        model_action = self.actor(state, self.action_size).detach().numpy() * self.action_range
        action = model_action + self.ou2.sample() * self.action_range
        # action = np.reshape(action,[1,self.action_size])
        return action

    def update_target_model(self):
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((deepcopy(state), action, reward, deepcopy(next_state), done))


    def _get_sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def train(self):
        minibatch = np.array(self._get_sample(self.batch_size)).transpose()

        states = np.vstack(minibatch[0])
        actions = np.vstack(minibatch[1])
        rewards = np.vstack(minibatch[2])
        next_states = np.vstack(minibatch[3])
        dones = np.vstack(minibatch[4].astype(int))

        rewards = torch.Tensor(rewards)
        dones = torch.Tensor(dones)
        actions = torch.Tensor(actions)
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)

        # critic update
        self.critic_optimizer.zero_grad()

        next_actions = self.actor_target(next_states, self.action_size)
        pred = self.critic(states, actions)
        next_pred = self.critic_target(next_states, next_actions)

        target = rewards + (1 - dones) * self.gamma * next_pred
        critic_loss = F.mse_loss(pred, target)
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        self.actor_optimizer.zero_grad()
        pred_actions = self.actor(states, self.action_size)
        actor_loss = self.critic(states, pred_actions).mean()
        actor_loss = -actor_loss
        actor_loss.backward()
        self.actor_optimizer.step()


def main():  # True

    import os
    buffer_dir = "./buffer_original_reward3"
    results_dir = "./results"
    SaveModel_dir = "./SaveModel"
    if not os.path.exists(buffer_dir):
        os.makedirs(buffer_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(SaveModel_dir):
        os.makedirs(SaveModel_dir)

    start_time = time.time()
    np.random.seed(1337)

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
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

    #agent.load_state_dict(torch.load('./SaveModel/3DDPG_save_original_reward2.pth'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    replay_buffer = utils.ReplayBuffer(state_dim=obs_size, action_dim=action_size, device=device)

    buffer_name = "test_buffer"

    recent_reward = deque(maxlen=100)
    frame = 0
    total_reward = []
    print("Torcs env start")
    total_step=0
    for e in count(1):  # range(args.episode):
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

        raw_state = raw_state.reshape(1,raw_state.shape[0])
        state = state.reshape(1, state.shape[0])

        while not done:
            total_step +=1
            step += 1
            frame += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)

            ob, reward, done, info, rs = env.step(action[0])
            next_state = np.hstack(
                (ob.angle, ob.track, ob.trackPos, ob.speedY, ob.speedX, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
            raw_next_state = np.hstack(
            (rs['angle'], rs['track'], rs['trackPos'], rs['speedY'], rs['speedX'],
             rs['speedZ'], rs['wheelSpinVel'], rs['rpm']))

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action[0], reward, next_state, done)
            if step>300:
                done = True

            #replay_bufferbuffer.add(state,action,next_state,reward,done,)
            replay_buffer.add(state, action, next_state, reward, done, raw_state, raw_next_state)

            score += reward
            raw_state = raw_next_state
            raw_state =raw_state.reshape(1,raw_state.shape[0])
            state = next_state
            state = state.reshape(1, state.shape[0])

            if frame > agent.batch_size:
                agent.train()
                agent.update_target_model()

            if done :
                total_reward.append(score)
                if total_step % 10000:
                    torch.save(agent.state_dict(), './SaveModel/3DDPG_save_original_reward3' + str(total_step) + '.pth')
                    plt.plot(list(range(len(total_reward))), total_reward, c="b", lw=3, ls="--")
                    plt.xlabel("Episode")
                    plt.ylabel("Reward")
                    plt.title("Reward Graph")
                    fig = plt.gcf()
                    # plt.show()
                    fig.savefig('./SaveModel/reward_graph22.png')
                    # plt.close()

                    #replay_buffer.save(f"./buffer_original_reward3/{buffer_name}")
                    print("buffer saved, total step :", total_step)

                recent_reward.append(score)

                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "   steps:", step,
                      "    recent reward:", np.mean(recent_reward))


            if (total_step == 100000):
                replay_buffer.save(f"./buffer_original_reward3/{buffer_name}")
                print("buffer saved, total step :", total_step)
                end_time = time.time()
                total_time = start_time - end_time
                print("training ended")
                print("total time : ", total_time)

                break
            if step>300:
                break

        if (total_step == 100000):
            break
                # if the mean of scores of last 10 episode is bigger than 400
                # stop training

    end_time = time.time()
    total_time = start_time - end_time
    print("training ended")
    print("total time : ", total_time)



if __name__ == '__main__':
    main()





