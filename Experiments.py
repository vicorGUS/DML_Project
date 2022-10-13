import time
import matplotlib
import numpy as np
from Helper import LearningCurvePlot
from Create_Dataset import create_Loaders


def create_plot(num_epochs, loss_fn, learning_rate,
                weight_decay, model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _, train_losses, train_accs, val_losses, val_accs = training_loop(
        model, optimizer, loss_fn, train_loader, val_loader, num_epochs=num_epochs, print_every=None)
    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    return train_losses, train_accs, val_losses, val_accs


def experiment():
    n_epochs = 50
    data = np.load('Melspectrogram_new.npz')

    # Standard settings
    learning_rate = 1e-4
    weight_decay = 1e-5
    batch_size = 32
    model = BIRDCNN()


    ### Experiment 1
    print('Learning rate experiment')
    Plot_1 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the learning rate', metrics=loss)
    Plot_2 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the learning rate', metrics=accuracy)

    # Settings
    learning_rates = [1e-5, 1e-4, 1e-3]

    for lr in learning_rates:
        train_loader, val_loader, weights = Create_Loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = create_plot(
            n_epochs, loss_fn, lr, weight_decay, model, train_loader, val_loader)
        Plot_1.add_curve(train_losses[::10], label='Training loss, lr = {}'.format(lr))
        Plot_1.add_curve(val_losses, label='Validation loss, lr = {}'.format(lr))
        Plot_2.add_curve(train_accs, label='Training accuracy, lr = {}'.format(lr))
        Plot_2.add_curve(val_accs, label='Validation accuracy, lr = {}'.format(lr))

    Plot1.save('Learning_rate_loss.png')
    Plot2.save('Learning_rate_accuracy.png')

    ### Experiment 2
    print('Weight decay experiment')
    Plot_1 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the weight decay', metrics=loss)
    Plot_2 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the weight decay', metrics=accuracy)

    # Settings
    weight_decays = [0, 1e-5, 1e-3]

    for wd in weight_decays:
        train_loader, val_loader, weights = Create_Loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = create_plot(
            n_epochs, loss_fn, learning_rate, wd, model, train_loader, val_loader)
        Plot_1.add_curve(train_losses, label='Training loss, wd = {}'.format(wd))
        Plot_1.add_curve(val_losses, label='Validation loss, wd = {}'.format(wd))
        Plot_2.add_curve(train_accs, label='Training accuracy, wd = {}'.format(wd))
        Plot_2.add_curve(val_accs, label='Validation accuracy, wd = {}'.format(wd))

    Plot1.save('Weight_decay_loss.png')
    Plot2.save('Weight_decay_accuracy.png')

    ### Experiment 3
    print('Batch size experiment')
    Plot_1 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the batch size', metrics=loss)
    Plot_2 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the batch size', metrics=accuracy)

    # Settings
    batch_sizes = [16, 32, 64]

    for bs in batch_sizes:
        train_loader, val_loader, weights = Create_Loaders(data, bs)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = create_plot(
            n_epochs, loss_fn, learning_rate, weight_decay, model, train_loader, val_loader)
        Plot_1.add_curve(train_losses, label='Training loss, batch size = {}'.format(bs))
        Plot_1.add_curve(val_losses, label='Validation loss, batch size = {}'.format(bs))
        Plot_2.add_curve(train_accs, label='Training accuracy, batch size = {}'.format(bs))
        Plot_2.add_curve(val_accs, label='Validation accuracy, batch size = {}'.format(bs))

    Plot1.save('Batch_size_loss.png')
    Plot2.save('Batch_size_accuracy.png')

    ### Experiment 4
    print('Ablation study')
    Plot_1 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the model architecture', metrics=loss)
    Plot_2 = LearningCurvePlot(
        title=r'Audio identification: Effect of changing the model architecture', metrics=accuracy)

    # Settings
    models = [, , ]

    for model_var in models:
        train_loader, val_loader, weights = Create_Loaders(data, batch_size)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
        train_losses, train_accs, val_losses, val_accs = create_plot(
            n_epochs, loss_fn, learning_rate, weight_decay, model_var, train_loader, val_loader)
        Plot_1.add_curve(train_losses, label='Training loss, {}'.format(model))
        Plot_1.add_curve(val_losses, label='Validation loss, {}'.format(model))
        Plot_2.add_curve(train_accs, label='Training accuracy, {}'.format(model))
        Plot_2.add_curve(val_accs, label='Validation accuracy, {}'.format(model))

    Plot1.save('Ablation_study_loss.png')
    Plot2.save('Ablation_study_accuracy.png')

if __name__ == "__main__":
    experiment()