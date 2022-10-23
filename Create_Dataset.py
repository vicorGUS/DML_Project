from sklearn.model_selection import train_test_split
from Helper import CustomTensorDataset, log_clipped
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch


## Creates data loader from spectrograms, given the batch size and spectrogram widths

def create_loaders(data, batch_size, spec_width=196):
    label = [i // 30 for i in range(2730)]
    input = [data[i] for i in data]
    log_input = [log_clipped(sample) for sample in input]
    new_input = []
    new_label = []
    for n, sample in enumerate(log_input):
        n_splits = sample.shape[1] // spec_width
        for split in range(n_splits):
            new_input.append(sample[:, int((sample.shape[1] - spec_width * (n_splits - split)) / 2):int(
                (sample.shape[1] - spec_width * (n_splits - 2 - split)) / 2)])
            new_label.append(label[n])
    weights = [1 / new_label.count(i) for i in range(91)]

    X_train, X_val, y_train, y_val = train_test_split(new_input, new_label, test_size=0.2, random_state=8,
                                                      stratify=new_label)

    Data_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = CustomTensorDataset(tensors=[X_train, y_train], transform=Data_transforms)
    val_data = CustomTensorDataset(tensors=[X_val, y_val], transform=Data_transforms)
    n_workers = (8 if torch.cuda.is_available()
                 else 0)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=n_workers)

    return train_loader, val_loader, weights
