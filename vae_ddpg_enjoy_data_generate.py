from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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

import numpy as np
s_prime = np.load("./buffer_original_reward3/test_buffer_next_state.npy")
train_loader = torch.utils.data.DataLoader(
    s_prime,
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    s_prime,
    batch_size=args.batch_size, shuffle=True)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(29, 300)
        self.fc11 = nn.Linear(300, 600)
        self.fc21 = nn.Linear(600, 29) # mu
        self.fc22 = nn.Linear(600, 29) # var
        self.fc3 = nn.Linear(29, 300)
        self.fc33 = nn.Linear(300, 600)
        self.fc4 = nn.Linear(600, 29)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h11 = F.relu(self.fc11(h1))
        return self.fc21(h11), self.fc22(h11)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h33 = F.relu(self.fc33(h3))
        return torch.sigmoid(self.fc4(h33))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 29))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, 29))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.float()
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)  # data = (128,1,28,28)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    torch.save(model.state_dict(), './SaveModel/VAE_DDPG.pth')

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device).float()
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if i == 0:
                with torch.no_grad():
                    sample = model.forward(data)[0].cpu()
                    np.save("generated_sample_data.npy", data.cpu().numpy())
                    np.save("generated_data.npy", sample.numpy())


            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 29)[:n]])
                # save_image(comparison.cpu(),
                #          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)



            # save_image(sample.view(64, 1, 29),
            #            'results/sample_' + str(epoch) + '.png')