import torch
from torch import nn


def d_loss_func(real_output, fake_output, device):
    return -torch.mean(real_output) + torch.mean(fake_output)

def g_loss_func(fake_output, device):
    return -torch.mean(fake_output)

generator = Generator(
    args.latent_dim,
    args.gen_dim
    args.seq_len, 
    vocab_size
)

discriminator = Discriminator(
    args.seq_len, 
    vocab_size
)

class Generator(nn.Module):
    def __init__(self, latent_dim, gen_dim, seq_len, vocab_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(latent_dim, gen_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(gen_dim * 8),
            nn.ReLU(True),
            # state size: (gen_dim*8) x 4
            nn.ConvTranspose1d(gen_dim * 8, gen_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(gen_dim * 4),
            nn.ReLU(True),
            # state size: (gen_dim*4) x 8
            nn.ConvTranspose1d(gen_dim * 4, gen_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(gen_dim * 2),
            nn.ReLU(True),
            # state size: (gen_dim*2) x 16
            nn.ConvTranspose1d(gen_dim * 2, gen_dim, 4, 2, 1, bias=False),
            nn.BatchNorm1d(gen_dim),
            nn.ReLU(True),
            # state size: (gen_dim) x 32
            nn.ConvTranspose1d(gen_dim, vocab_size, 4, 2, 1, bias=False),
            nn.BatchNorm1d(vocab_size),
            nn.ReLU(True),
            # state size: (vocab_size) x 64
            nn.ConvTranspose1d(vocab_size, vocab_size, 4, 2, 1, bias=False),
            nn.BatchNorm1d(vocab_size),
            nn.ReLU(True),
            # state size: (vocab_size) x 128
            nn.ConvTranspose1d(vocab_size, vocab_size, 4, 2, 1, bias=False),
            nn.BatchNorm1d(vocab_size),
            nn.ReLU(True),
            # state size: (vocab_size) x 256
            nn.ConvTranspose1d(vocab_size, vocab_size, 4, 2, 1, bias=False),
            nn.Softmax(dim=1) 
            # output size: (vocab_size) x 300
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, vocab_size, gen_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (vocab_size) x 300
            nn.Conv1d(vocab_size, gen_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (gen_dim) x 150
            nn.Conv1d(gen_dim, gen_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(gen_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (gen_dim*2) x 75
            nn.Conv1d(gen_dim * 2, gen_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(gen_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (gen_dim*4) x 37
            nn.Conv1d(gen_dim * 4, gen_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(gen_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (gen_dim*8) x 18
            nn.Conv1d(gen_dim * 8, 1, 18, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)