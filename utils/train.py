#TODO

import torch
import torchvision
from torch import nn
from models import discriminator, global_discriminator, local_discriminator
from models import generator
from train_funcs import _weights_init, display_progress, configure_optimizers


# These configurations are from paper
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = nn.L1Loss() 
lambda_recon = 100

n_epochs = 20

display_step = 10
batch_size = 1
lr = 0.002
target_size = 256
device = 'cuda'   # GPU is VVIMP

local_input_shape = (3, 128, 128)
global_input_shape = (3, 256, 256)
arc = 2


# discriminator training step
def _disc_step(gen, disc, adversarial_criterion, real_images, conditioned_images):
        fake_images = gen(conditioned_images).detach()
        fake_images = nn.functional.interpolate(fake_images, size=128)
        fake_logits = disc((fake_images, conditioned_images))

        real_logits = disc((real_images, conditioned_images))

        fake_loss = adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2


# generator training step
def _gen_step(gen, disc, adversarial_criterion, recon_criterion, real_images, conditioned_images, lambda_recon):
        fake_images = gen(conditioned_images).detach()
        fake_images = nn.functional.interpolate(fake_images, size=128)
        disc_logits = disc((fake_images, conditioned_images))
        adversarial_loss = adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = recon_criterion(fake_images, real_images)

        return adversarial_loss + lambda_recon * recon_loss


def train_GAN(gen, disc, dataloader, max_epochs, adversarial_criterion, recon_criterion, lr=0.002, lambda_recon=100):    
 
        """
        -Main training function for SketchGAN
        -TODO : Add Classfier loss

        -adversarial_loss = nn.BCEWithLogitsLoss()

        -reconstruction_loss = nn.L1Loss()

        """
        optimizer_idx = 1
        optimizer_D, optimizer_G = configure_optimizers(gen, disc, lr)
        gen = gen.to(device)
        disc = disc.to(device)
        for epoch in range(max_epochs):    
            gen.train()
            disc.train()
            for batch_idx, (real, condition) in enumerate(dataloader): 
                real = real.to(device)
                condition = condition.to(device)
                loss = None
                if optimizer_idx == 0:
                    d_loss = _disc_step(gen, disc, adversarial_criterion, real, condition)
                    optimizer_D.zero_grad()
                    d_loss.backward(retain_graph=True)
                    optimizer_D.step()
                    loss = d_loss
                    optimizer_idx = 1
                elif optimizer_idx == 1:
                    g_loss = _gen_step(gen, disc, adversarial_criterion, recon_criterion, real, condition, lambda_recon)
                    optimizer_G.zero_grad()
                    g_loss.backward(retain_graph=True)
                    optimizer_G.step()
                    loss = g_loss
                    optimizer_idx = 0
                
                if batch_idx==0 and optimizer_idx==1:
                    fake = gen(condition).detach()
                    display_progress(condition[0], fake[0], real[0], loss)