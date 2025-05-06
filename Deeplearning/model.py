import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64*28*28, 64),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        output = self.model(x)
        return output


# 测试模型
if __name__ == '__main__':
    net = Net()
    input = torch.ones((64, 3, 32, 32))
    output = net(input)
    print(output.shape)
