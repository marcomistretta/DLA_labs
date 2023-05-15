import torch.nn as nn
from torch.nn.functional import relu


class MLP(nn.Module):
    def __init__(self, sizes, in_channels):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_channels, sizes[0]), nn.ReLU()]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes=10, depth=20):
        super(CNN, self).__init__()
        print("CNN depth:", depth)
        layers = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResCNN(nn.Module):
    def __init__(self, num_classes=10, depth=20):
        super(ResCNN, self).__init__()
        print("ResCNN depth:", depth)
        layers = [nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers.append(BasicBlock(64, 64))
        self.features = nn.Sequential(*layers)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.avgPool(self.features(x))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.stride != 1 or identity.shape[1] != out.shape[1]:
            identity = nn.Conv2d(identity.shape[1], out.shape[1], kernel_size=1, stride=self.stride, bias=False)(
                identity)
            identity = nn.BatchNorm2d(out.shape[1])(identity)
        out += identity
        out = self.relu(out)
        return out
