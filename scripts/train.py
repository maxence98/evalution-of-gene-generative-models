import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from utils.utils import DotDict
from utils.dna import DNADataset
from utils import dna
from trainer import GANExperiment

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Arguments
args = DotDict({
    #experiment setting
    "experiment_name": "snwgp5res0401",
    "model": "SNWGP5Res",
    "custom_weight": True,
    "discriminator_ratio": 5, #The ratio of the discriminator to generator updates
    "num_epochs": 1500, #300, 
    "sample_size": 2, #How many batches of sample latents for testing, 0 for no sample
    "sample_every_n_epochs": 30, #6,
    "save_every_n_epochs": 500, #100,

    #hyperparameter
    "learning_rate": 1e-5,
    "adamBetas": (0.5, 0.999), #default(0.5, 0.999)

    #data
    "data_loc": "../data/human",
    "vocab": "ATGC", 

    #model setting (fixed)
    "latent_dim": 100,
    "seq_len": 300,

    #residual model setting
    "gen_dim": 100,
    "res_layers": 2,
    "lmbda": 10., #Lipschitz penalty hyperparameter

    #training setting
    "random_seed": 42,
    "batch_size": 64,
    "log_dir": "../runs",
})

# Set random seed for reproducibility
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

# Data Loader    
data = dna.load(
    args.data_loc, 
    seq_len=args.seq_len,
    vocab=args.vocab,
    random_seed=args.random_seed,
)
vocab_size = len(args.vocab)

train_dataset = DNADataset(data["train"])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset = DNADataset(data["val"])
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize generator and discriminator
use_gradient_penalty = False
if args.model == "WGP5Res":
    from models.WGP5Res import *
    use_gradient_penalty = True
    generator = Generator(
        args.latent_dim,
        args.gen_dim,
        args.seq_len, 
        args.res_layers,
        vocab_size
    ).to(device)

    discriminator = Discriminator(
        args.gen_dim,
        args.seq_len, 
        args.res_layers,
        vocab_size
    ).to(device)
elif args.model == "SNWGP5Res":
    from models.SNWGP5Res import *
    use_gradient_penalty = True
    generator = Generator(
        args.latent_dim,
        args.gen_dim,
        args.seq_len, 
        args.res_layers,
        vocab_size
    ).to(device)

    discriminator = Discriminator(
        args.gen_dim,
        args.seq_len, 
        args.res_layers,
        vocab_size
    ).to(device)
elif args.model == "WGP":
    from models.WGP import *
    use_gradient_penalty = True
    generator = Generator(
        args.latent_dim,
        args.seq_len, 
        vocab_size
    ).to(device)

    discriminator = Discriminator(
        args.seq_len, 
        vocab_size
    ).to(device)
elif args.model == "WGPm":
    from models.WGP import *
    use_gradient_penalty = True
    generator = Generator(
        args.latent_dim,
        args.seq_len, 
        vocab_size
    ).to(device)

    discriminator = Discriminator(
        args.seq_len, 
        vocab_size
    ).to(device)
elif args.model == "WGANGP2D":
    from models.WGANwithoutRes import *
    use_gradient_penalty = True
    generator = Generator(
        args.latent_dim,
        args.gen_dim,
        args.seq_len, 
        args.res_layers,
        vocab_size
    ).to(device)

    discriminator = Discriminator(
        args.seq_len, 
        vocab_size
    ).to(device)
elif args.model == "DCGAN":
    from models.DC5Res import *
    use_gradient_penalty = False
    generator = Generator(
        args.latent_dim,
        args.gen_dim,
        args.seq_len, 
        args.res_layers,
        vocab_size
    ).to(device)

    discriminator = Discriminator(
        args.gen_dim,
        args.seq_len, 
        args.res_layers,
        vocab_size
    ).to(device)
elif args.model == "SNWGP":
    from models.SNWGP import *
    use_gradient_penalty = True
    generator = Generator(
        args.latent_dim,
        args.seq_len, 
        vocab_size
    ).to(device)

    discriminator = Discriminator(
        args.seq_len, 
        vocab_size
    ).to(device)

# Customly initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

if args.custom_weight:
    generator.apply(weights_init)
    discriminator.apply(weights_init)

trainer = GANExperiment(
    generator, 
    discriminator, 
    train_loader, 
    val_loader, 
    d_loss_func,
    g_loss_func,
    args,
    device,
    use_gradient_penalty,
)

trainer.train()
