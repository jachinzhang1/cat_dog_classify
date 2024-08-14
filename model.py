import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.linear_1 = nn.Linear(28 * 28 * 256, 128)
        self.linear_2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [2,3,224,224]->[2,32,112,112]
        x = self.conv1_1(x)
        # [2,32,224,224]->[2,64,112,112]
        x = self.conv1_2(x)
        # [2,64,112,112]->[2,64,112,112]
        x = self.conv2_1(x)
        # [2,64,112,112]->[2,128,56,56]
        x = self.conv2_2(x)
        # [2,64,112,112]->[2,128,56,56]
        x = self.conv3_1(x)
        # [2,64,112,112]->[2,256,28,28]
        x = self.conv3_2(x)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        out = self.sigmoid(self.linear_2(x))
        return out


if __name__ == '__main__':
    model = Model()
    x = torch.randn(2, 3, 224, 224)
    print(model)
    print(model(x).shape)
    print(model(x))
