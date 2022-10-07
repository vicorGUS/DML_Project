import torch
import torch.nn as nn
from Network import BIRDCNN
from Train import training_loop
from Create_Data import train_loader, val_loader

model = BIRDCNN()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs=1, print_every=None)