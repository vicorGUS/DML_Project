import torch
import torch.nn as nn
from Network import BIRDCNN, DEEPCNN, BIRD2CNN
from Train import training_loop
from Create_Data import train_loader, val_loader, weights

model = BIRD2CNN()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))

training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs=50, print_every=None)