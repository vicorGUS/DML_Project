import numpy as np
import pydub
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset


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

y = np.zeros(2730)
for i in range(91):
    y[i*30:(i+1)*30] = i

y = torch.from_numpy(y)
y = F.one_hot(y.to(torch.int64), num_classes=91)

A = np.empty([len(metadata['file_name']), 129, 200])

"""
for i, name in enumerate(metadata['file_name']):
    sr, x = read(f'archive/xeno-canto-ca-nv/{name}')
    if np.ndim(x) > 1:
        x = x[:, 0]
    x = np.concatenate((100 * np.random.normal(size=max_length - int(len(x)/2)), x, 100 * np.random.normal(size=max_length - int(len(x)/2))))
    Sxx = spectrogram(x, fs=sr)[2]
    Sxx_max = np.where(Sxx == np.max(Sxx))
    Sxx = np.log(Sxx[:, Sxx_max[1][0] - 100:Sxx_max[1][0] + 100])
    print(Sxx.shape, i)
    A[i] = Sxx

A_reshaped = A.reshape(A.shape[0], -1)

np.savetxt("Spectrograms.txt", A_reshaped)"""

A_reshaped = np.loadtxt("Spectrograms.txt")

A_reshaped = np.nan_to_num(A)

A = A_reshaped.reshape([len(metadata['file_name']), 1, 129, 200])

X_train, X_val, y_train, y_val = train_test_split(torch.from_numpy(A), y,
                                                  test_size=0.2, random_state=8)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=64)