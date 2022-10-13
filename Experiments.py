import time
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from Helper import LearningCurvePlot
from Train import training_loop
from Create_Dataset import create_loaders
from Network import StandardCNN, ShallowCNN, DeepCNN


def train(num_epochs, loss_fn, learning_rate,
          weight_decay, model, train_loader, val_loader):
    now = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _, train_losses, train_accs, val_losses, val_accs = training_loop(
        model, optimizer, loss_fn, train_loader, val_loader, num_epochs=num_epochs, print_every=None)
    print('Running one setting takes {:.1f} minutes'.format((time.time() - now) / 60))
    return train_losses, train_accs, val_losses, val_accs


def experiment():
    n_epochs = 2
    data = np.load('Melspectrogram_new.npz')

    # Standard settings
    learning_rate = 1e-4
    weight_decay = 1e-5
    batch_size = 32
    model = StandardCNN()

    ### Experiment 1
    print('Learning rate experiment')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the learning rate', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the learning rate', metrics='accuracy')

    # Settings
    learning_rates = [1e-5, 1e-4, 1e-3]

    for lr in learning_rates:
        train_loader, val_loader, weights = create_loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, lr, weight_decay, model, train_loader, val_loader)
        batch_per_epoch = int(len(train_losses) / len(val_losses))
        Plot_loss.add_curve(train_losses[::batch_per_epoch], label='Training loss, lr = {:.1e}'.format(lr))
        Plot_loss.add_curve(val_losses, label='Validation loss, lr = {:.1e}'.format(lr))
        Plot_accuracy.add_curve(train_accs[::batch_per_epoch], label='Training accuracy, lr = {:.1e}'.format(lr))
        Plot_accuracy.add_curve(val_accs, label='Validation accuracy, lr = {:.1e}'.format(lr))

    Plot_loss.save('Learning_rate_loss.png')
    Plot_accuracy.save('Learning_rate_accuracy.png')

    ### Experiment 2
    print('Weight decay experiment')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the weight decay', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the weight decay', metrics='accuracy')

    # Settings
    weight_decays = [0, 1e-5, 1e-3]

    for wd in weight_decays:
        train_loader, val_loader, weights = create_loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, learning_rate, wd, model, train_loader, val_loader)
        batch_per_epoch = int(len(train_losses) / len(val_losses))
        Plot_loss.add_curve(train_losses[::batch_per_epoch], label='Training loss, wd = {:.1e}'.format(wd))
        Plot_loss.add_curve(val_losses, label='Validation loss, wd = {:.1e}'.format(wd))
        Plot_accuracy.add_curve(train_accs[::batch_per_epoch], label='Training accuracy, wd = {:.1e}'.format(wd))
        Plot_accuracy.add_curve(val_accs, label='Validation accuracy, wd = {:.1e}'.format(wd))

    Plot_loss.save('Weight_decay_loss.png')
    Plot_accuracy.save('Weight_decay_accuracy.png')

    ### Experiment 3
    print('Batch size experiment')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the batch size', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the batch size', metrics='accuracy')

    # Settings
    batch_sizes = [16, 32, 64]

    for bs in batch_sizes:
        train_loader, val_loader, weights = create_loaders(data, bs)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, learning_rate, weight_decay, model, train_loader, val_loader)
        batch_per_epoch = int(len(train_losses) / len(val_losses))
        Plot_loss.add_curve(train_losses[::batch_per_epoch], label='Training loss, batch size = {}'.format(bs))
        Plot_loss.add_curve(val_losses, label='Validation loss, batch size = {}'.format(bs))
        Plot_accuracy.add_curve(train_accs[::batch_per_epoch], label='Training accuracy, batch size = {}'.format(bs))
        Plot_accuracy.add_curve(val_accs, label='Validation accuracy, batch size = {}'.format(bs))

    Plot_loss.save('Batch_size_loss.png')
    Plot_accuracy.save('Batch_size_accuracy.png')

    ### Experiment 4
    print('Ablation study')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the model architecture', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the model architecture', metrics='accuracy')

    # Settings
    models = [StandardCNN(), ShallowCNN(), DeepCNN()]

    for model_var in models:
        train_loader, val_loader, weights = create_loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, learning_rate, weight_decay, model_var, train_loader, val_loader)
        batch_per_epoch = int(len(train_losses) / len(val_losses))
        Plot_loss.add_curve(train_losses[::batch_per_epoch], label='Training loss, {}'.format(model))
        Plot_loss.add_curve(val_losses, label='Validation loss, {}'.format(model))
        Plot_accuracy.add_curve(train_accs[::batch_per_epoch], label='Training accuracy, {}'.format(model))
        Plot_accuracy.add_curve(val_accs, label='Validation accuracy, {}'.format(model))

    Plot_loss.save('Ablation_study_loss.png')
    Plot_accuracy.save('Ablation_study_accuracy.png')


if __name__ == "__main__":
    experiment()
