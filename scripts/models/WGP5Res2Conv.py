import torch
from torch import nn

def d_loss_func(real_output, fake_output, device):
    return -torch.mean(real_output) + torch.mean(fake_output)

def g_loss_func(fake_output, device):
    return -torch.mean(fake_output)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filter_width=5):
        super(ResidualBlock, self).__init__()
        
        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(
                in_channels, 
                in_channels, 
                filter_width, 
                stride=1, 
                padding=(filter_width - 1)//2,
                bias=True
            )
        )
  
    def forward(self, x):
        return x + 0.3 * self.main(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, gen_dim, seq_len, res_layers, vocab_size):
        super(Generator, self).__init__()
        self.gen_dim = gen_dim
        self.seq_len = seq_len

        self.linear = nn.Linear(latent_dim, gen_dim * seq_len)
        self.resblock = nn.Sequential(*[ResidualBlock(gen_dim) for _ in range(res_layers)])
        self.final_conv = nn.Conv1d(
            gen_dim, 
            vocab_size, 
            1, 
            stride=1, 
            padding=0,
            bias=True
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, z):
        out = self.linear(z)
        out = torch.reshape(out, (-1, self.gen_dim, self.seq_len))
        out = self.resblock(out)
        out = self.final_conv(out)
        out = self.softmax(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, seq_len, vocab_size):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Downsampling layers
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(self.vocab_size, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
        )

        # Output layer
        self.fc = nn.Linear(128*self.seq_len//4, 1) # 2 times of stride of 2

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.shape[0], -1)  # Flatten feature maps
        return self.fc(out)