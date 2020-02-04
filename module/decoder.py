import functools

import torch
from torch import nn
import torch.nn.functional as F

from inplace_abn.bn import InPlaceABNSync

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


class SEModule(nn.Module):
    """Squeeze and Extraction module"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class DecoderHead(nn.Module):

    def __init__(self, in_dim, out_dim, d_rate=[12, 24, 36]):
        super(DecoderHead, self).__init__()
        self.b0 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                                BatchNorm2d(out_dim), nn.ReLU(inplace=False))

        self.b1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[0], dilation=d_rate[0], bias=False),
                                BatchNorm2d(out_dim), nn.ReLU(inplace=False))
        self.b2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[1], dilation=d_rate[1], bias=False),
                                BatchNorm2d(out_dim), nn.ReLU(inplace=False))
        self.b3 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3,
                                          padding=d_rate[2], dilation=d_rate[2], bias=False),
                                BatchNorm2d(out_dim), nn.ReLU(inplace=False))
        self.b4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, bias=False),
                                BatchNorm2d(out_dim), nn.ReLU(inplace=False))

        self.project = nn.Sequential(nn.Conv2d(out_dim * 5, out_dim, kernel_size=1, padding=0, bias=False),
                                     InPlaceABNSync(out_dim),
                                     nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                     InPlaceABNSync(out_dim))

    def forward(self, x):
        h, w = x.size()[2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = F.interpolate(self.b4(x), size=(h, w), mode='bilinear', align_corners=True)
        out = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1)
        return self.project(out)


class Node1(nn.Module):

    def __init__(self, node1_cls):
        super(Node1, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv4 = nn.Conv2d(256, node1_cls, kernel_size=1, padding=0, dilation=1, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xt, xm, xl):
        _, _, h, w = xm.size()
        xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        _, _, th, tw = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        output = self.conv4(x)
        return output


class Node2(nn.Module):
    def __init__(self, node2_cls):
        super(Node2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   nn.Conv2d(256, node2_cls, kernel_size=1, padding=0, stride=1, bias=True))

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha * skip
        output = self.conv1(xfuse)
        return output


class Node3(nn.Module):
    def __init__(self, node3_cls):
        super(Node3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16),
                                   nn.Conv2d(256, node3_cls, kernel_size=1, padding=0, stride=1, bias=True))

        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.beta * skip
        output = self.conv1(xfuse)
        return output


class HierarchyDecoder(nn.Module):
    def __init__(self, num_classes):
        super(HierarchyDecoder, self).__init__()
        self.layer5 = DecoderHead(2048, 512)
        self.layer_n1 = Node1(node1_cls=num_classes)
        self.layer_n2 = Node2(node2_cls=3)
        self.layer_n3 = Node3(node3_cls=2)

        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])
        seg_node1 = self.layer_n1(seg, x[1], x[0])
        seg_node2 = self.layer_n2(seg, x[1])
        seg_node3 = self.layer_n3(seg, x[1])
        return [seg_node1, seg_node2, seg_node3, x_dsn]
