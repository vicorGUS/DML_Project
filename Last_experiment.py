import time
import matplotlib as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from Helper import LearningCurvePlot
from Train import training_loop
from Create_Dataset import create_loaders
from Network import StandardCNN
from sklearn.metrics import confusion_matrix



def train(num_epochs, loss_fn, learning_rate,
          weight_decay, model, train_loader, val_loader):
    now = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, train_losses, train_accs, val_losses, val_accs = training_loop(
        model, optimizer, loss_fn, train_loader, val_loader, num_epochs=num_epochs, print_every=None)
    print('Running one setting takes {:.1f} minutes'.format((time.time() - now) / 60))
    return train_losses, train_accs, val_losses, val_accs, model


def experiment():
    n_epochs = 1
    data = np.load('Melspectrogram_new.npz')

    # Standard settings
    learning_rate = 1e-4
    weight_decay = 1e-5
    batch_size = 24

    ### Experiment 1
    print('Final model')
    Plot_loss = LearningCurvePlot(
        title=r'Audio identification loss: Final model', metrics='loss')
    Plot_accuracy = LearningCurvePlot(
        title=r'Audio identification accuracy: Final model', metrics='accuracy')

    # Settings
    model = StandardCNN()
    train_loader, val_loader, weights, inputs, labels = create_loaders(data, batch_size)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights))
    train_losses, train_accs, val_losses, val_accs, model = train(
        n_epochs, loss_fn, learning_rate, weight_decay, model, train_loader, val_loader)
    Plot_loss.add_curve(train_losses, label='Training data')
    Plot_loss.add_curve(val_losses, label='Validation data')
    Plot_accuracy.add_curve(train_accs, label='Training data')
    Plot_accuracy.add_curve(val_accs, label='Validation data')
    Plot_loss.save('Final_model_loss.png')
    Plot_accuracy.save('Final_model_accuracy.png')



    pred = []
    cm = confusion_matrix(labels, pred)
    cm = cm.astype('float64')
    for i in range(cm.shape[0]):
        cm[i, :] = cm[i, :] / sum(cm[i, :])
        
    
    metadata = pd.read_csv('archive/xeno-canto_ca-nv_index.csv')
    plt.figure(figsize=(10, 10))
    plt.imshow(cm)
    plt.xticks(range(len(metadata['english_cname'].unique())), metadata['english_cname'].unique(), rotation='vertical')
    plt.yticks(range(len(metadata['english_cname'].unique())), metadata['english_cname'].unique())
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()






if __name__ == "__main__":
    experiment()