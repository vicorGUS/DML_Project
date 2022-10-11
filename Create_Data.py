import librosa
import matplotlib.pyplot as plt
import numpy as np
import pydub
from scipy.signal import spectrogram
from librosa import feature
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


metadata = pd.read_csv('archive/xeno-canto_ca-nv_index.csv')
length_sec = metadata['duration_seconds']
max_length = np.max(length_sec * 25000)

y = torch.zeros(2730)

for i in range(91):
    y[i*30:(i + 1)*30] = i

"""
A = np.empty([len(metadata['file_name']), 128, 196])
B = np.empty([len(metadata['file_name']), 128, 196])

for i, name in enumerate(metadata['file_name']):
    sr, x = read(f'archive/xeno-canto-ca-nv/{name}')
    if np.ndim(x) > 1:
        x = x[:, 0]
    x = np.concatenate((np.random.uniform(0, 1, size=max_length - int(len(x)/2)), x, np.random.uniform(0, 1, size=max_length - int(len(x)/2))))
    x_max = np.argmax(x)
    Sxx = feature.melspectrogram(y=x[x_max-50000:x_max+50000], sr=sr)
    Stft = librosa.stft(y=x[x_max-50000:x_max+50000], n_fft=255, hop_length=511)
    print(i)
    A[i] = Sxx
    B[i] = np.real(Stft)
    assert Sxx.shape == (128, 196), print(Sxx.shape)

A_reshaped = A.reshape(A.shape[0], -1)
B_reshaped = B.reshape(B.shape[0], -1)

#np.savetxt("Melspectrogram.txt", A_reshaped)
#np.savetxt("Stft.txt", B_reshaped)"""

A_reshaped = np.loadtxt("Melspectrogram.txt")
B_reshaped = np.loadtxt("Stft.txt")

A = A_reshaped.reshape([len(metadata['file_name']), 128, 196])
B = B_reshaped.reshape([len(metadata['file_name']), 128, 196])

C = np.zeros([len(metadata['file_name']), 2, 128, 196])
C[:, 0] = A
C[:, 1] = B

"""for i in range(0, 2730, 40):
    plt.imshow(A[i, 0])
    plt.show()"""

X_train, X_val, y_train, y_val = train_test_split(C, y,
                                                  test_size=0.2, random_state=8)

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((np.mean(A, axis=0), np.mean(B, axis=0)),
                                                                             (np.std(A, axis=0)), np.std(B, axis=0))])

train_data = CustomTensorDataset(tensors=[X_train, y_train], transform=None)
val_data = CustomTensorDataset(tensors=[X_val, y_val], transform=None)

train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=32)