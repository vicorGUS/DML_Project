import numpy as np
import librosa
import pandas as pd
from librosa import feature
from Helper import read


def data_from_mp3(Create_stft=False):
    metadata = pd.read_csv('archive/xeno-canto_ca-nv_index.csv')
    mel = []
    shorttft = []
    for i, name in enumerate(metadata['file_name']):
        sr, x = read(f'archive/xeno-canto-ca-nv/{name}', normalized=True)
        if np.ndim(x) > 1:
            x = x[:, 0]
        Sxx = feature.melspectrogram(y=x, sr=sr)
        mel.append(Sxx)
        if Create_stft == True:
            Stft = np.real(librosa.stft(y=x, n_fft=255))
            shorttft.append(Stft)
    np.savez("Melspectrogram_new.npz", *mel)
    if Create_stft:
        np.savez("Stft_new.npz", *shorttft)

    print("Done!")
