import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import math


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


''''--------------------------------------------Feature Extractor----------------------------------------------------'''


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3

    for v in cfg:
        if v == 'M':
            # down-sample rate is 2
            layers += [nn.MaxPool2d(2)]
        else:
            # size will not change dut to padding
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            # in channel of next layer is the out channel of current layer
            in_channels = v

    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()

        # feature extractor
        self.features = features
        # output size is (7, 7)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        '''weight initilization'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1e-2)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class extractor(nn.Module):
    def __init__(self, pretrained: bool):
        super(extractor, self).__init__()

        vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
        if pretrained:
            vgg16_bn.load_state_dict(torch.load('.vgg16_bn-6c64b313.pth'))

        # feature extractor,
        # note that here, only use this part of backbone in forward progress,
        # it doesn't contain the head of backbone
        self.features = vgg16_bn.features

    def forward(self, x):
        out = []

        for m in self.features:
            x = m(x)

            if isinstance(m, nn.MaxPool2d):
                out.append(x)

        # 1/4, 1/8, 1/16, 1/32
        return out[1:]


'''---------------------------------------------Feature Fusion-------------------------------------------------------'''


class merge(nn.Module):
    def __init__(self):
        super(merge, self).__init__()

        # 定义conv stage _1
        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        # 定义conv stage_2
        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # 定义conv stage_3
        self.conv5 = nn.Conv2d(192, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        # 定义conv stage_4
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        '''weight initialization'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1/32 -> 1/16
        y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        # channel = 512 x 2 = 1024
        y = torch.cat((y, x[2]), 1)
        '''channel=128'''
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        # 1/16 -> 1/8
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        # channel = 128 + 256 = 384
        y = torch.cat((y, x[1]), 1)
        '''channel=64'''
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        # 1/8 -> 1/4
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        # channel = 64 + 128 = 192
        y = torch.cat((y, x[0]), 1)
        '''channel=32'''
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        # 1/4, channel=32
        y = self.relu7(self.bn7(self.conv7(y)))

        return y


'''-------------------------------------------------Map Output-------------------------------------------------------'''


class output(nn.Module):
    def __init__(self, scope=512):
        super(output, self).__init__()

        # 请定义self.conv1,用于输出score map
        # 建议代码：self.conv1 = nn.Conv2d(, , )
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()

        # 请定义self.conv2,用于输出d1,d2,d3,d4 map
        # 建议代码：self.conv2 = nn.Conv2d(,,)
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()

        # 请定义self.conv3,用于输出angle_map
        # 建议代码：self.conv3 = nn.Conv2d(, , )
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()

        self.scope = scope

        '''weight initilization'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shape: (N, 1, 1/4h, 1/4w)
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        # shape: (N, 5, 1/4h, 1/4w)
        geo = torch.cat((loc, angle), 1)

        return score, geo


class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()

        self.extractor = extractor(pretrained)
        self.merge = merge()
        self.output = output()

    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))


if __name__ == '__main__':
    m = EAST(pretrained=False)
    x = torch.randn(1, 3, 256, 256)
    score, geo = m(x)

    print(score.shape)
    print(geo.shape)
