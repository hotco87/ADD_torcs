import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

HIDDEN1 = 300
HIDDEN2 = 600

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, HIDDEN1)
		self.l2 = nn.Linear(HIDDEN1, HIDDEN2)
		self.l3 = nn.Linear(HIDDEN2, action_dim)

		self.max_action = max_action

	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x))
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, num_heads):
		super(Critic, self).__init__()
		self.num_heads = num_heads

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, HIDDEN1)
		self.l2 = nn.Linear(HIDDEN1, HIDDEN2)
		self.l3 = nn.Linear(HIDDEN2, num_heads)

		# # Q2 architecture
		# self.l4 = nn.Linear(state_dim + action_dim, 400)
		# self.l5 = nn.Linear(400, 300)
		# self.l6 = nn.Linear(300, 1)


	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		return self.l3(x1)

	def Q_value(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1.mean(dim=-1, keepdim=True)


class REM(object):
	"""Random Ensemble Mixture."""
	def __init__(self, state_dim, action_dim, max_action, lr=1e-3, num_heads=200): #2
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = Critic(state_dim, action_dim, num_heads).to(device)
		self.critic_target = Critic(state_dim, action_dim, num_heads).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

		self.max_action = max_action


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

		for it in range(iterations):

			# Sample replay buffer
			x, u, y, r, d = replay_buffer.sample(batch_size)#x, y, u, r, d = replay_buffer.sample(batch_size)
			x = x.cpu()
			y = y.cpu()
			u = u.cpu()
			r = r.cpu()
			d = d.cpu()
			state = torch.FloatTensor(x).to(device)
			action = torch.FloatTensor(u).to(device)
			next_state = torch.FloatTensor(y).to(device)
			done = torch.FloatTensor(1 - d).to(device)
			reward = torch.FloatTensor(r).to(device)

			# Select action according to policy and add clipped noise
			noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			num_heads = self.critic.num_heads
			target_Q_heads = self.critic_target(next_state, next_action)
			alpha = torch.rand((num_heads, 1))
			alpha /= alpha.sum(dim=0)
			alpha = alpha.to(device)
			target_Q = torch.matmul(target_Q_heads, alpha)
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimates
			current_Q_heads = self.critic(state, action)
			current_Q = torch.matmul(current_Q_heads, alpha)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Delayed policy updates
			if it % policy_freq == 0:

				# Compute actor loss
				actor_loss = -(self.critic.Q_value(state, self.actor(state)).mean())

				# Optimize the actor
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				# Update the frozen target agent
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
