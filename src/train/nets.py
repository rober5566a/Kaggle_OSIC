import torch
import torch.nn as nn


# class Net(nn.Module):
#     def __init__(self, input_dim=10, output_dim=1):
#         super(Net, self).__init__()

#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(256, output_dim)
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x


class Net(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


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
