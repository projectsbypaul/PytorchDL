import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # N, 3, 32, 32
        x = F.relu(self.conv1(x))  # -> N, 32, 30, 30
        x = self.pool(x)  # -> N, 32, 15, 15
        x = F.relu(self.conv2(x))  # -> N, 64, 13, 13
        x = self.pool(x)  # -> N, 64, 6, 6
        x = F.relu(self.conv3(x))  # -> N, 64, 4, 4
        x = torch.flatten(x, 1)  # -> N, 1024
        x = F.relu(self.fc1(x))  # -> N, 64
        x = self.fc2(x)  # -> N, 10
        return x



