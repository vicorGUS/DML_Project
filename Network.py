import torch
import torch.nn as nn
import torch.nn.functional as F


class BIRDCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(5 * 10 * 32, 128)
        self.fc2 = nn.Linear(128, 91)

    def forward(self, input_batch):
        print(input_batch.shape)
        x = self.conv1(input_batch)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

