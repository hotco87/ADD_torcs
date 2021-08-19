import numpy as np
import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        self.ou = OrnsteinUhlenbeckActionNoise(theta=0.60, sigma=0.30, mu=0.0, action_dim=self.action_size)
        self.ou2 = OrnsteinUhlenbeckActionNoise(theta=1.0, sigma=0.10, mu=0.0, action_dim=self.action_size)
        self.ou3 = OrnsteinUhlenbeckActionNoise(theta=1.0, sigma=0.50, mu=-0.1, action_dim=self.action_size)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # initialize target model
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state, enjoy=False):
        state = torch.from_numpy(state).float()
        model_action = self.actor(state, self.action_size).detach().numpy() * self.action_range

        action = model_action if enjoy else model_action + self.ou2.sample() * self.action_range

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