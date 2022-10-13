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
    n_epochs = 20
    data = np.load('Melspectrogram_new.npz')

    # Standard settings
    learning_rate = 1e-4
    weight_decay = 1e-5
    batch_size = 32

    ### Experiment 1
    print('Learning rate experiment')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification loss: Effect of changing the learning rate', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification accuracy: Effect of changing the learning rate', metrics='accuracy')

    # Settings
    learning_rates = [1e-5, 1e-4, 1e-3]

    for lr in learning_rates:
        model = StandardCNN()
        train_loader, val_loader, weights = create_loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, lr, weight_decay, model, train_loader, val_loader)
        Plot_loss.add_curve(train_losses, label='Training, lr = {:.0e}'.format(lr).replace("e-0", "e-"))
        Plot_loss.add_curve(val_losses, label='Validation, lr = {:.0e}'.format(lr).replace("e-0", "e-"))
        Plot_accuracy.add_curve(train_accs, label='Training, lr = {:.0e}'.format(lr).replace("e-0", "e-"))
        Plot_accuracy.add_curve(val_accs, label='Validation, lr = {:.0e}'.format(lr).replace("e-0", "e-"))

    Plot_loss.save('Learning_rate_loss.png')
    Plot_accuracy.save('Learning_rate_accuracy.png')

    ### Experiment 2
    print('Weight decay experiment')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification loss: Effect of changing the weight decay', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification accuracy: Effect of changing the weight decay', metrics='accuracy')

    # Settings
    weight_decays = [0, 1e-5, 1e-3]

    for wd in weight_decays:
        model = StandardCNN()
        train_loader, val_loader, weights = create_loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, learning_rate, wd, model, train_loader, val_loader)
        if wd == 0:
            Plot_loss.add_curve(train_losses, label='Training, wd = {:d}'.format(wd))
            Plot_loss.add_curve(val_losses, label='Validation, wd = {:d}'.format(wd))
            Plot_accuracy.add_curve(train_accs, label='Training, wd = {:d}'.format(wd))
            Plot_accuracy.add_curve(val_accs, label='Validation, wd = {:d}'.format(wd))
        else:
            Plot_loss.add_curve(train_losses, label='Training, wd = {:.0e}'.format(wd).replace("e-0", "e-"))
            Plot_loss.add_curve(val_losses, label='Validation, wd = {:.0e}'.format(wd).replace("e-0", "e-"))
            Plot_accuracy.add_curve(train_accs, label='Training, wd = {:.0e}'.format(wd).replace("e-0", "e-"))
            Plot_accuracy.add_curve(val_accs, label='Validation, wd = {:.0e}'.format(wd).replace("e-0", "e-"))

    Plot_loss.save('Weight_decay_loss.png')
    Plot_accuracy.save('Weight_decay_accuracy.png')

    ### Experiment 3
    print('Batch size experiment')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification loss: Effect of changing the batch size', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification accuracy: Effect of changing the batch size', metrics='accuracy')

    # Settings
    batch_sizes = [16, 32, 64]

    for bs in batch_sizes:
        model = StandardCNN()
        train_loader, val_loader, weights = create_loaders(data, bs)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, learning_rate, weight_decay, model, train_loader, val_loader)
        Plot_loss.add_curve(train_losses, label='Training, batch size = {}'.format(bs))
        Plot_loss.add_curve(val_losses, label='Validation, batch size = {}'.format(bs))
        Plot_accuracy.add_curve(train_accs, label='Training, batch size = {}'.format(bs))
        Plot_accuracy.add_curve(val_accs, label='Validation, batch size = {}'.format(bs))

    Plot_loss.save('Batch_size_loss.png')
    Plot_accuracy.save('Batch_size_accuracy.png')

    ### Experiment 4
    print('Ablation study')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification loss: Effect of changing the model architecture', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification accuracy: Effect of changing the model architecture', metrics='accuracy')

    # Settings
    models = [StandardCNN(), ShallowCNN(), DeepCNN()]
    model_names = ['Standard CNN', 'Shallow CNN', 'Deep CNN']

    for i, model_var in enumerate(models):
        train_loader, val_loader, weights = create_loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = train(
            n_epochs, loss_fn, learning_rate, weight_decay, model_var, train_loader, val_loader)
        Plot_loss.add_curve(train_losses, label='Training, {}'.format(model_names[i]))
        Plot_loss.add_curve(val_losses, label='Validation, {}'.format(model_names[i]))
        Plot_accuracy.add_curve(train_accs, label='Training, {}'.format(model_names[i]))
        Plot_accuracy.add_curve(val_accs, label='Validation, {}'.format(model_names[i]))

    Plot_loss.save('Ablation_study_loss.png')
    Plot_accuracy.save('Ablation_study_accuracy.png')


if __name__ == "__main__":
    experiment()
