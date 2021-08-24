from __future__ import print_function
import argparse

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader

from vrae import VRAE

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

s = np.load('./buffer_original_reward3/test_buffer_state.npy')
a = np.load('./buffer_original_reward3/test_buffer_action.npy')
r = np.load('./buffer_original_reward3/test_buffer_reward.npy')
s_ = np.load('./buffer_original_reward3/test_buffer_next_state.npy')
not_done = np.load('./buffer_original_reward3/test_buffer_not_done.npy')

idx_done = np.array(np.where(not_done.reshape(-1) != 1))
idx_epi_start = np.insert(idx_done + 1, 0, 0)
idx_epi_end = np.append(idx_done, not_done.size)
epi_idxes = zip(idx_epi_start, idx_epi_end)

min_sequence_len = np.min(np.diff(idx_done, 1))
max_sequence_len = np.max(np.diff(idx_done, 1))

# episodes = []
# for start, finish in epi_idxes:
#     current_episode = s[start:finish]
#     padded_current_episode = np.concatenate(
#         (current_episode, np.zeros((max_sequence_len - current_episode.shape[0], 29))), axis=0)
#     episodes.append(padded_current_episode)
# episodes = np.stack(episodes)

seq_len = 8
episodes = []
for start, finish in epi_idxes:
    current_episode = s[start:finish]
    frag = [np.array(current_episode[i:i + seq_len]) for i in range(len(current_episode) - seq_len)]
    episodes.append(np.stack(frag))
episodes = np.concatenate(episodes)

train_dataset = TensorDataset(torch.from_numpy(episodes))
test_dataset = TensorDataset(torch.from_numpy(episodes))

##############################################################
hidden_size = 90
hidden_layer_depth = 1
latent_length = 64
batch_size = 32
learning_rate = 0.0005
n_epochs = 50
dropout_rate = 0.2
optimizer = 'Adam'  # options: ADAM, SGD
cuda = True  # options: True, False
print_every = 200
clip = True  # options: True, False
max_grad_norm = 5
loss = 'MSELoss'  # options: SmoothL1Loss, MSELoss
block = 'LSTM'  # options: LSTM, GRU
##############################################################

sequence_length = episodes.shape[1]
number_of_features = episodes.shape[2]

model = VRAE(sequence_length=sequence_length,
             number_of_features=number_of_features,
             hidden_size=hidden_size,
             hidden_layer_depth=hidden_layer_depth,
             latent_length=latent_length,
             batch_size=batch_size,
             learning_rate=learning_rate,
             n_epochs=n_epochs,
             dropout_rate=dropout_rate,
             optimizer=optimizer,
             cuda=cuda,
             print_every=print_every,
             clip=clip,
             max_grad_norm=max_grad_norm,
             loss=loss,
             block=block)

model.fit(train_dataset)
model.save('SaveModel/vrae_s_model.pth')

x_decoded = model.fit_transform(test_dataset)

print('asdf')


