import torch
import torch.nn as nn

def d_loss_func(real_output, fake_output, device):
    return nn.BCELoss(real_output, torch.ones(real_output.shape[0], 1).to(device))

def g_loss_func(fake_output, device):
    return nn.BCELoss(fake_output, torch.zeros(fake_output.shape[0], 1).to(device))

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
    def __init__(self, gen_dim, seq_len, res_layers, vocab_size):
        super(Discriminator, self).__init__()
        self.gen_dim = gen_dim
        self.seq_len = seq_len

        self.first_conv = nn.Conv1d(
            vocab_size, 
            gen_dim, 
            1, 
            stride=1, 
            padding=0,
            bias=True
        )
        self.resblock = nn.Sequential(*[ResidualBlock(gen_dim) for _ in range(res_layers)])
        self.linear = nn.Linear(gen_dim * seq_len, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        
        out = self.first_conv(z)
        out = self.resblock(out)
        out = torch.reshape(out, (-1, self.seq_len * self.gen_dim))
        out = self.linear(out)
        score = self.sigmoid(out)
        return score