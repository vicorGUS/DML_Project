import librosa
import matplotlib.pyplot as plt
import numpy as np
import pydub
from librosa import feature
from librosa.display import specshow
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return len(self.tensors[0])


def read(f, normalized=False):
    a = pydub.AudioSegment.from_mp3(f)
    b = np.array(a.get_array_of_samples())
    if a.channels == 2:
        b = b.reshape([-1, 2])
    if normalized:
        return a.frame_rate, np.float32(b) / 2 ** 15
    else:
        return a.frame_rate, b


def log_clipped(a):
    return np.log(np.clip(a, .0000001, a.max()))


metadata = pd.read_csv('archive/xeno-canto_ca-nv_index.csv')

y = torch.zeros(2730)
for i in range(91):
    y[i * 30:(i + 1) * 30] = i

"""A = []
B = []

for i, name in enumerate(metadata['file_name']):
    sr, x = read(f'archive/xeno-canto-ca-nv/{name}', normalized=True)
    if np.ndim(x) > 1:
        x = x[:, 0]
    Sxx = feature.melspectrogram(y=x, sr=sr)
    #Stft = np.real(librosa.stft(y=x, n_fft=255))
    A.append(Sxx)
    #B.append(Stft)
    print(i)

np.savez("Melspectrogram_new.npz", *A)"""
#np.savez("Stft_new.npz", *B)

data = np.load('Melspectrogram_new.npz')

A = [data[i] for i in data]

#A = np.loadtxt("Melspectrogram.txt")
#B = np.loadtxt("Stft.txt")

#A = A_reshaped.reshape([len(metadata['file_name']), 128, 196])
#B = B_reshaped.reshape([len(metadata['file_name']), 128, 196])

#C = np.zeros([len(metadata['file_name']), 2, 128, 196])
#C[:, 0] = A
#C[:, 1] = B

"""for i in range(2730):
    plt.imshow(np.log(A[i]))
    plt.show()"""

X_train, X_val, y_train, y_val = train_test_split(A, y,
                                                  test_size=0.2, random_state=8, stratify=y)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.RandomCrop(size=(128, 196), pad_if_needed=True)])

train_data = CustomTensorDataset(tensors=[X_train, y_train], transform=transforms)
val_data = CustomTensorDataset(tensors=[X_val, y_val], transform=transforms)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=32)