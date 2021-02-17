"""
Date: 15/02/2021

Description:
            Variational Autoencoder with Pytorch
            Initial code with Pytorch in order to understand Pytorch and compare with TensorFlow

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


print("Pytorch Version:", torch.__version__)  # 1.7.1+cu101
print("GPU available:", torch.cuda.is_available())

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# MNIST dataset
dataset = datasets.MNIST(root='', train=True, transform=transforms.ToTensor(), download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=True)


class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=4, stride=2, padding=1),   # N, 32, 14, 14
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),   # N, 64, 7, 7
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),  # N, 128, 3,3
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(1152, 32)
        self.fc2 = nn.Linear(1152, 32)
        self.fc3 = nn.Linear(32, 1024)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(1024, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        xr = self.fc3(z)
        xr = self.decoder(xr)
        return xr

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_r = self.decode(z)
        return x_r, mu, logvar


def loss_fn(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, size_average=False)
    #BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss_t = BCE + KLD
    return loss_t, BCE, KLD


#model = Net()
model = Net().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


epochs = 2
for epoch in range(epochs):
    print(f"Epochs:{epoch}/{epochs}")
    for idx, (images, _) in enumerate(data_loader):
        images = images.to(device)

        recon_images, mu, logvar = model(images)
        loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss, idx)
