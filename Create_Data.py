import numpy as np
import pydub
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F
import torch


def read(f, normalized=False):
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2 ** 15
    else:
        return a.frame_rate, y


metadata = pd.read_csv('archive/xeno-canto_ca-nv_index.csv')

#print(metadata['english_cname'].unique())

y = np.zeros(2730)
for i in range(91):
    y[i*30:(i+1)*30] = i

y = torch.from_numpy(y)
y = F.one_hot(y.to(torch.int64), num_classes=91)

length_sec = metadata['duration_seconds']

max_length = np.max(length_sec * 50000)

A = []

for name in metadata['file_name']:
    sr, x = read(f'archive/xeno-canto-ca-nv/{name}')
    if np.ndim(x) > 1:
        x = x[:, 0]
    x = np.concatenate((x, 100 * np.random.normal(size=max_length - len(x))))
    Sxx = spectrogram(x, fs=sr)[2]
    Sxx_max = np.where(Sxx == np.max(Sxx))
    Sxx = np.log(Sxx[:, Sxx_max[1][0] - 100:Sxx_max[1][0] + 100])
    A.append(Sxx)

X_train, X_val, y_train, y_val = train_test_split(A, y,
                                                  test_size=0.2, random_state=8)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_data, batch_size=25, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=25)