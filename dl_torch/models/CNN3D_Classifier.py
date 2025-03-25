import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the 3D CNN model
class CNN3D_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3D_Classifier, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 15, 15, 15]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7, 7]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 128, 3, 3, 3]

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Logits (raw scores)
        x = F.softmax(x, dim=1) # softmax for probabilities

        return x


class CNN3D_Classifier_MLPlus(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN3D_Classifier_MLPlus, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):


        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 15, 15, 15]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7, 7]
        x = self.pool(F.relu(self.conv3(x)))  # [batch, 128, 3, 3, 3]

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)  # Logits (raw scores)
        x = F.softmax(x, dim=1)  # softmax for probabilities

        return x


def main() -> None:
    Test = CNN3D_Classifier(10)
    print(Test)

if __name__ == '__main__':
    main()