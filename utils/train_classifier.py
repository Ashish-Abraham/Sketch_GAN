from models import classifier
import torch.nn as nn
import torch.cuda
import torchvision
from torchvision.datasets import ImageFolder
from torch import optim
from torchvision.transforms import ToTensor, Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip
import os
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils import tensorboard as tb

train_data = ImageFolder(dir, transform=Compose([Resize([225, 225]), RandomHorizontalFlip(p=0.2), RandomVerticalFlip(p=0.2), ToTensor()]))

model = classifier.Sketch_A_Net(in_channels=1)
model = model.to('cuda')

# configurations from paper
n_epochs = 230
batch_size = 128
lr = 0.001
device = 'cuda'
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_size = int(0.8 * len(train_data))
test_size = len(train_data) - train_size
train_dataset, test_dataset = random_split(train_data, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=True, drop_last=True)
val_loader = DataLoader(test_dataset, batch_size=batch_size*2, pin_memory=True, num_workers=4, shuffle=True, drop_last=True)

# training function
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, val_loader, print_interval=10 ):
    writer = tb.SummaryWriter('./logs')
    count = 0
    for e in range(n_epochs):
        for i, (X, Y) in enumerate(train_loader):
            # Binarizing 'X'
            X[X < 1.] = 0.; X = 1. - X
            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            optimizer.zero_grad()

            output = model(X)
            loss = loss_fn(output.squeeze(), Y)
            
            if i % print_interval == 0:
                print(f'[Training] {i}/{e}/{n_epochs} -> Loss: {loss.item()}')
                writer.add_scalar('train-loss', loss.item(), count)
            
            loss.backward()
            optimizer.step()

            count += 1

        correct, total = 0, 0
        for i, (X, Y) in enumerate(val_loader):
            # Binarizing 'X'
            X[X < 1.] = 0.; X = 1. - X

            if torch.cuda.is_available():
                X, Y = X.cuda(), Y.cuda()

            output = model(X)
            _, predicted = torch.max(output, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            accuracy = (correct / total) * 100
            if accuracy>85.0:
              torch.save(model.state_dict(), 'best_model_weights.pth')
        
        print(f'[Testing] -/{e}/{n_epochs} -> Accuracy: {accuracy} %')
        writer.add_scalar('test-accuracy', accuracy/100., e)