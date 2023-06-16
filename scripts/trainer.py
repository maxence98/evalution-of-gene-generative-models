import os
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class GANExperiment:
    def __init__(
        self, 
        generator, 
        discriminator, 
        train_loader,
        val_loader,
        d_loss_func,
        g_loss_func,
        args,
        device,
        use_gradient_penalty
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.d_loss_func = d_loss_func
        self.g_loss_func = g_loss_func
        self.args = args
        self.device = device
        self.use_gradient_penalty = use_gradient_penalty
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.learning_rate, betas=args.adamBetas)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=args.learning_rate, betas=args.adamBetas)

    def compute_gradient_penalty(self, real_batch, fake_batch):
        eps = torch.rand(real_batch.size(0), 1, 1).to(self.device)
        interpolated_batch = (eps * real_batch + (1 - eps) * fake_batch).requires_grad_(True)
        interpolated_output = self.discriminator(interpolated_batch)
        gradients = torch.autograd.grad(
            outputs=interpolated_output,
            inputs=interpolated_batch,
            grad_outputs=torch.ones_like(interpolated_output).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient_penalty = ((gradients.view(gradients.shape[0], -1).norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def makeSampleLatents(self):
        return [
            torch.normal(0, 1, size=(self.args.batch_size, self.args.latent_dim)) 
            for _ in range(self.args.sample_size)
        ]

    def train(self):

        log_folder = os.path.join(self.args.log_dir, self.args.experiment_name)
        writer = SummaryWriter(log_folder)
        if self.args.sample_size != 0:
            sample_folder = os.path.join(log_folder, "samples")
            os.makedirs(sample_folder, exist_ok=True)
            sampleLatents = self.makeSampleLatents()
        if self.args.save_every_n_epochs != 0:
            checkpoint_folder = os.path.join(log_folder, "checkpoints")
            os.makedirs(checkpoint_folder, exist_ok=True)
        
        iteration = 0
        for epoch in range(self.args.num_epochs):
            self.generator.train()
            self.discriminator.train()
            for i, batch in enumerate(self.train_loader):
                # Train the discriminator
                real_batch = batch.to(self.device)
                batch_size = real_batch.size(0)
                z = torch.normal(0, 1, size=(batch_size, self.args.latent_dim)).to(self.device)
                fake_batch = self.generator(z)

                self.d_optimizer.zero_grad()
                real_output = self.discriminator(real_batch)
                fake_output = self.discriminator(fake_batch.detach())
                d_loss = self.d_loss_func(real_output, fake_output, self.device) 
                if self.use_gradient_penalty:
                    d_loss += self.args.lmbda * self.compute_gradient_penalty(real_batch, fake_batch)
                d_loss.backward()
                self.d_optimizer.step()

                # Train the generator
                if (i+1) % self.args.discriminator_ratio == 0:
                    writer.add_scalar('D_loss', d_loss.item(), iteration)

                    self.g_optimizer.zero_grad()
                    z = torch.normal(0, 1, size=(batch_size, self.args.latent_dim)).to(self.device)
                    fake_batch = self.generator(z)
                    fake_output = self.discriminator(fake_batch)
                    g_loss = self.g_loss_func(fake_output, self.device) 
                    g_loss.backward()
                    self.g_optimizer.step()
                    writer.add_scalar('G_loss', g_loss.item(), iteration)
                    iteration += 1
                
                    #print(iteration, d_loss.item(), g_loss.item())

            # Validation
            self.discriminator.eval()
            self.generator.eval()
            d_losses = []
            for val_batch in self.val_loader:
                val_batch = val_batch.to(self.device)
                z = torch.normal(0, 1, size=(batch_size, self.args.latent_dim)).to(self.device)
                fake_batch = self.generator(z)

                val_output = self.discriminator(val_batch)
                fake_output = self.discriminator(fake_batch.detach())   

                d_loss_val = self.d_loss_func(val_output, fake_output, self.device) 
                d_losses.append(d_loss_val.item())
            val_loss = np.mean(d_losses)
            writer.add_scalar('val_loss', val_loss, iteration)

            # save sample generated instances
            if epoch % self.args.sample_every_n_epochs == 0 and self.args.sample_size != 0:
                samples = []
                for sample_z in sampleLatents:
                    sample_z = sample_z.to(self.device)
                    with torch.no_grad():   
                        sample_batch = self.generator(sample_z)
                    samples.append(sample_batch.cpu().numpy())
                np.save(os.path.join(sample_folder, f"iter_{iteration}.npy"), np.array(samples))
                del sample_z
            
            # save model
            if self.args.save_every_n_epochs != 0 and epoch % self.args.save_every_n_epochs == 0:
                checkpoint = {
                    'epoch': epoch,
                    'g': self.generator.state_dict(),
                    'd': self.discriminator.state_dict(),
                    'g_opt': self.g_optimizer.state_dict(),
                    'd_opt': self.d_optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(checkpoint_folder, f"iter_{iteration}.pth"))

            del real_batch
            del z
            del val_batch
            torch.cuda.empty_cache()

        writer.close()
