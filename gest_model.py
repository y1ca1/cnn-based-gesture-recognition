import torch
import torch.nn as nn
import torch.nn.functional as F
from our_parser import get_config
config = get_config()
import math

ThreeLayer = [32, 64, 128]
TwoLayer = [32, 64]


class ThreeLayerConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ThreeLayer[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(ThreeLayer[0]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(ThreeLayer[0], ThreeLayer[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(ThreeLayer[1]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(ThreeLayer[1], ThreeLayer[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(ThreeLayer[2]),
            nn.ReLU(),
            # nn.MaxPool2d(2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )

        self.fc = nn.Linear(ThreeLayer[2] * 16 * 16, config['num_class'])
        self.dropout = nn.Dropout(p=0.8)

    @staticmethod
    def flatten(x):
        N = x.shape[0]  # read in N, C, H, W
        return x.contiguous().view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, x):
        scores = None
        pass
        conv = self.conv_layer(x)
        scores = self.fc(self.flatten(conv))
        scores = self.dropout(scores)
        return scores


class TwoLayerConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        pass
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, TwoLayer[0], kernel_size=5, padding=2),
            nn.BatchNorm2d(TwoLayer[0]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(TwoLayer[0], TwoLayer[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(TwoLayer[1]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_layer = nn.Sequential(
            self.layer1,
            self.layer2,
            # self.layer3
        )

        self.fc = nn.Linear(TwoLayer[1] * 16 * 16, config['num_class'])
        self.dropout = nn.Dropout(p=0.8)

    @staticmethod
    def flatten(x):
        N = x.shape[0]  # read in N, C, H, W
        return x.contiguous().view(N, -1)  # "flatten" the C * H * W values into a single vector per image

    def forward(self, x):
        scores = None
        pass
        conv = self.conv_layer(x)
        scores = self.fc(self.flatten(conv))
        scores = self.dropout(scores)
        return scores


def test_ThreeLayerConvNet():
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % config['gpu_id'] if use_cuda else "cpu")
    dtype = torch.float32
    x = torch.zeros((64, 3, 64, 64), dtype=dtype)  # minibatch size 64, image size [3, 64, 64]
    model = ThreeLayerConvNet(config)
    scores = model(x)
    print(scores.size())  # you should see [64, 11]


test_ThreeLayerConvNet()
