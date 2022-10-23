import numpy as np
import librosa
import pandas as pd
from librosa import feature
from Helper import read


## Converts mp3 files into spectrograms

def data_from_mp3():
    metadata = pd.read_csv('archive/xeno-canto_ca-nv_index.csv')
    mel = []
    for i, name in enumerate(metadata['file_name']):
        sr, x = read(f'archive/xeno-canto-ca-nv/{name}', normalized=True)
        if np.ndim(x) > 1:
            x = x[:, 0]
        Sxx = feature.melspectrogram(y=x, sr=sr)
        mel.append(Sxx)
    np.savez("Melspectrogram.npz", *mel)
    print("Done!")


if __name__ == "__main__":
    data_from_mp3()
