from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1000000, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(29, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, 29)
        self.log_std = nn.Linear(750, 29)

        self.d1 = nn.Linear(58, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, 29)

        self.latent_dim = 58
        self.device = device

    def forward(self, state):
        z = F.relu(self.e1(state))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        # a = F.relu(self.d1(state))
        a = F.relu(self.d2(a))
        return self.d3(a)

model = VAE().to(device)
model.load_state_dict(torch.load('./SaveModel/VAE_DDPG.pth'))
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def data_generate():
    import numpy as np
    s_prime = np.load("./buffer_original_reward3/test_buffer_next_state.npy")
    s_prime = torch.FloatTensor(s_prime).to(device)
    for i in range(10):
        with torch.no_grad():
            sample = model.forward(s_prime)[0].cpu()
            np.save("./buffer_original_reward3/vae_buffer_next_state"+str(i+1)+".npy", sample.cpu().numpy())

if __name__ == "__main__":
    data_generate()
