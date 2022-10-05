import numpy as np
from scipy.fft import fft
import pydub
import matplotlib.pyplot as plt
import pandas as pd


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

length_sec = metadata['duration_seconds']

max_length = np.max(length_sec * 50000)

A = []

for name in metadata['file_name']:
    sr, x = read(f'archive/xeno-canto-ca-nv/{name}')
    if np.ndim(x) > 1:
        x = x[:, 0]
    x = np.concatenate((x, 100 * np.random.normal(size=max_length - len(x))))
    A.append(plt.specgram(x, Fs=sr))
