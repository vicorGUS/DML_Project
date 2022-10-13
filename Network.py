import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(5 * 9 * 32, 128)
        self.fc2 = nn.Linear(128, 91)

    def forward(self, input_batch):
        x = self.conv1(input_batch)
        x = self.pool(F.relu(self.conv2(x)))
        x = F.normalize(x)
        x = self.conv3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = F.normalize(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x


class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(13 * 22 * 32, 128)
        self.fc2 = nn.Linear(128, 91)

    def forward(self, input_batch):
        x = self.conv1(input_batch)
        x = self.pool(F.relu(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.conv2(x))
        x = F.normalize(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4)
        x = self.fc2(x)
        return x


class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=9, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(12 * 21 * 16, 91)

    def forward(self, input_batch):
        x = self.conv1(input_batch)
        x = self.pool(F.relu(x))
        x = F.relu(self.conv2(x))
        x = F.normalize(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        return x
