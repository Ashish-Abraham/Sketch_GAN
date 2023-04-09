"""
  - Train using Pytorch Lightning 
  - Improved Workflow
  - Training Code
    trainer = pl.Trainer(max_epochs, no_of_gpus)
    trainer.fit(sketchgan, dataloader)

  - Run the commands 
    ! pip install -q pytorch-lightning
    ! pip install config
    ! pip install GPUtil

"""
# TODO
"""
> define variable target_class by obtaining it from the dataset
"""

import torch
import torchvision
import pytorch_lightning as pl
from torch import nn
from models import discriminator, global_discriminator, local_discriminator
from models import generator, classifier
from train_funcs import _weights_init, display_progress, configure_optimizers, get_classifier_loss
from tqdm.auto import tqdm

# initialize classifier
classifier_model = classifier.Sketch_A_Net(in_channels=1)
classifier_model.load_state_dict(torch.load('model_weights.pth'))  # add path to classifier weights correctly


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



class SketchGAN(pl.LightningModule):

    def __init__(
                  self, in_channels, out_channels, batch_size,local_input_shape=(3,128,128), global_input_shape=(3,256,256), 
                  arc=2, learning_rate=0.002, lambda_recon=100, lambda_classifier=0.5, display_step=25, 
                ):

        super().__init__()
        self.save_hyperparameters()
        
        self.display_step = display_step
        self.gen = generator.Generator(in_channels, out_channels, batch_size)
        self.disc = discriminator.ContextDiscriminator(local_input_shape, global_input_shape, arc=2)

        # intializing weights
        self.gen = self.gen.apply(_weights_init)
        self.disc = self.disc.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def _gen_step(self, real_images, conditioned_images):
        # adversarial loss
        fake_images = self.gen(conditioned_images)
        fake_images = nn.functional.interpolate(fake_images, size=128)
        disc_logits = self.disc((fake_images, conditioned_images))
        adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))
        classifier_loss = get_classifier_loss(fake_images, target_class, classifier_model)   # target_class to be defined

        # reconstruction loss
        recon_loss = self.recon_criterion(fake_images, real_images)
        lambda_recon = self.hparams.lambda_recon
        lambda_classifier = self.hparams.lambda_classifier

        return adversarial_loss + lambda_recon * recon_loss + lambda_classifier*classifier_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.gen(conditioned_images).detach()
        fake_images = nn.functional.interpolate(fake_images, size=128)
        fake_logits = self.disc((fake_images, conditioned_images))

        real_logits = self.disc((real_images, conditioned_images))

        fake_loss = self.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.adversarial_criterion(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
        return disc_opt, gen_opt

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, condition = batch
        real = real.to(device)
        condition = condition.to(device)

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
            self.log('Discriminator Loss', loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
            self.log('Generator Loss', loss)
        
        if self.current_epoch%self.display_step==0 and batch_idx==0 and optimizer_idx==1:
            fake = self.gen(condition).detach()
            display_progress(condition[0], fake[0], real[0])
        return loss