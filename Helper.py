class LearningCurvePlot:

    def __init__(self, title=None, metrics=None):
        self.fig, self.ax = plt.subplots()
        if metrics == loss:
            self.ax.set_ylabel('Loss')
        elif metrics == accuracy:
            self.ax.set_ylabel('Accuracy')
        self.ax.set_xlabel('Episode')

        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label):
        self.ax.axhline(height, ls='--', c='k', label=label)

    def save(self, name='test.png'):
        self.ax.legend()
        self.fig.savefig(name, dpi=300)


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
