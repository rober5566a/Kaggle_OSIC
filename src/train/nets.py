import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_dim=10, input_channel=20, output_dim=1):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(),

            nn.Linear(16, 16),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(16 + 32, output_dim)
        )

    def forward(self, x, image):
        feature_01 = self.fc1(x)
        feature_02 = self.cnn(image).view(image.size(0), -1)

        feature = torch.cat([feature_01, feature_02], dim=1)
        out = self.fc2(feature)

        return out


class NetOI(nn.Module):
    def __init__(self, input_dim=10, input_channel=20, output_dim=1):
        super(NetOI, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.AdaptiveAvgPool2d((2, 2))
        )

        self.fc = nn.Sequential(
            nn.Linear(1024 + 10, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 512),
            nn.LeakyReLU(),

            nn.Linear(512, output_dim)
        )

    def forward(self, x, image):
        y = self.cnn(image).view(image.size(0), -1)
        # y = torch.cat([y, x[:, -1].unsqueeze(-1)], dim=1)
        y = torch.cat([y, x], dim=1)
        y = self.fc(y)

        return y


class NetSimple(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super(NetSimple, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
