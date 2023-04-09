import torch
import torchvision
from torch import nn
from models import discriminator, global_discriminator, local_discriminator
from models import generator
from models import classifier


# initialize weights
def _weights_init(m):
    m = m.to(device)  # use device = 'cuda' 
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        

# display training progress
def display_progress(cond, fake, real, gen_loss, figsize=(10,5), ):
    cond = cond.detach().cpu().permute(1, 2, 0)
    fake = fake.detach().cpu().permute(1, 2, 0)
    real = real.detach().cpu().permute(1, 2, 0)
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(cond)
    ax[2].imshow(fake)
    ax[1].imshow(real)
    plt.show()
    print("gen_loss: ",gen_loss)


# configure optimizers for training
def configure_optimizers(gen, disc, lr):
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    return disc_opt, gen_opt  

# get loss value from classifier
def get_classifier_loss(input, ground_truth, model):
    output = model(input)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, ground_truth)
    return loss  