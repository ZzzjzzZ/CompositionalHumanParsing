from module import *

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


class Stream(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(Stream, self).__init__()
        self.encoder = ResNet(block, layers)
        self.decoder = HierarchyDecoder(num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_model(num_classes=20):
    model = Stream(Bottleneck, [3, 4, 23, 3], num_classes)
    return model
