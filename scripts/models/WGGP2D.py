import torch
from torch import nn

def d_loss_func(real_output, fake_output):
    return -torch.mean(real_output) + torch.mean(fake_output)

def g_loss_func(fake_output):
    return -torch.mean(fake_output)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(self.latent_dim, 256*100)

        # Transposed convolutions to upsample the sequence
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 3), stride=(1, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 3), stride=(1, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(1, 3), stride=(1, 2), padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 256, 1, 100)  # batch_size, channel, H, W
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 3), stride=(1, 2), padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(256*50, 1),  # 50 is the resulting sequence length after the convolutions (400/2/2/2)
            #nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.shape[0], -1)  # Flatten the tensor
        return self.fc(out)