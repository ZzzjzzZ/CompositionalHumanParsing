import functools

import torch.nn as nn

from inplace_abn.bn import InPlaceABNSync

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


class Bottleneck(nn.Module):
    """Residual blocks"""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):
    """ResNet based on Bottleneck"""

    def __init__(self, block, layers, multi_grid=True):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, affine=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, affine=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128, affine=True)
        self.relu3 = nn.ReLU(inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # TODO: change the dilation rate
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=multi_grid)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=True))

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))

        self.inplanes = planes * block.expansion
        if multi_grid:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation * (2 ** i)))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]
