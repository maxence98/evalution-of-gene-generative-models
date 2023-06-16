import torch
from torch import nn

def d_loss_func(real_output, fake_output, device):
    return -torch.mean(real_output) + torch.mean(fake_output)

def g_loss_func(fake_output, device):
    return -torch.mean(fake_output)

class Generator(nn.Module):
    def __init__(self, latent_dim, seq_len, vocab_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.fc = nn.Linear(self.latent_dim, 100*self.seq_len)

        # Upsampling layers
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(100, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, vocab_size, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1)  # Softmax to output probabilities for each nucleotide
        )
    
    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 100, self.seq_len)  # Reshape to correct dimensions
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    def __init__(self, seq_len, vocab_size):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Downsampling layers
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(self.vocab_size, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            # nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
        )

        # Output layer
        self.fc = nn.Linear(32*self.seq_len//4, 1) # 2 times of stride of 2

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.shape[0], -1)  # Flatten feature maps
        return self.fc(out)