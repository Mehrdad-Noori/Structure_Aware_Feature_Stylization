import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from feature_stylization import MixStyle

def get_configs(arch='resnet50'):
    # True or False means whether to use BottleNeck

    if arch == 'resnet18':
        return [2, 2, 2, 2], False, 512
    elif arch == 'resnet34':
        return [3, 4, 6, 3], False, 512
    elif arch == 'resnet50':
        return [3, 4, 6, 3], True, 2048
    elif arch == 'resnet101':
        return [3, 4, 23, 3], True, 2048
    elif arch == 'resnet152':
        return [3, 8, 36, 3], True, 2048
    else:
        raise ValueError("Undefined model")


class ResNetAutoEncoderClassifier(nn.Module):

    def __init__(self, arch, num_class=7, reconstruction=True, state_dict_path=None, feature_stylization=False, p=0.5):
        super(ResNetAutoEncoderClassifier, self).__init__()
        configs, bottleneck, code_size = get_configs(arch)
        self.reconstruction = reconstruction

        self.encoder = ResNetEncoder(arch, feature_stylization=feature_stylization, p=p)

        if self.reconstruction:
            self.decoder = ResNetDecoder(configs=configs[::-1], bottleneck=bottleneck)
        self.classifier = Classifier(code_size, num_class)

        if state_dict_path:
            self.load_state_dict(torch.load(state_dict_path))
            self.eval()

    def forward(self, x):
        x_enc = self.encoder(x)
        x_class = self.classifier(x_enc)
        if self.reconstruction:
            x_dec = self.decoder(x_enc)
            return x_class, x_dec
        else:
            return x_class


class Classifier(nn.Module):
    def __init__(self, code_size, num_class):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(code_size, num_class)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)

        x = F.leaky_relu(x)
        x = F.dropout(x, 0.2)
        x = self.fc(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, arch, pretrained=True, feature_stylization=False, p=0.5):
        super(ResNetEncoder, self).__init__()
        if arch == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
        elif arch == 'resnet34':
            self.base_model = models.resnet34(pretrained=pretrained)
        elif arch == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
        elif arch == 'resnet101':
            self.base_model = models.resnet101(pretrained=pretrained)
        elif arch == 'resnet152':
            self.base_model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Undefined model")

        self.feature_stylization = feature_stylization
        self.p = p
        self.mixstyle = MixStyle(p=self.p, alpha=0.1)

        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool

        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if self.feature_stylization:
            x = self.mixstyle(x)
        x = self.layer2(x)
        if self.feature_stylization:
            x = self.mixstyle(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



class ResNetDecoder(nn.Module):

    def __init__(self, configs, bottleneck=False):
        super(ResNetDecoder, self).__init__()

        if len(configs) != 4:
            raise ValueError("Only 4 layers can be configued")

        if bottleneck:

            self.conv1 = DecoderBottleneckBlock(in_channels=2048, hidden_channels=512, down_channels=1024,
                                                layers=configs[0])
            self.conv2 = DecoderBottleneckBlock(in_channels=1024, hidden_channels=256, down_channels=512,
                                                layers=configs[1])
            self.conv3 = DecoderBottleneckBlock(in_channels=512, hidden_channels=128, down_channels=256,
                                                layers=configs[2])
            self.conv4 = DecoderBottleneckBlock(in_channels=256, hidden_channels=64, down_channels=64,
                                                layers=configs[3])


        else:

            self.conv1 = DecoderResidualBlock(hidden_channels=512, output_channels=256, layers=configs[0])
            self.conv2 = DecoderResidualBlock(hidden_channels=256, output_channels=128, layers=configs[1])
            self.conv3 = DecoderResidualBlock(hidden_channels=128, output_channels=64, layers=configs[2])
            self.conv4 = DecoderResidualBlock(hidden_channels=64, output_channels=64, layers=configs[3])

        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=7, stride=2, padding=3, output_padding=1,
                               bias=False),
        )

        self.gate = nn.Sigmoid()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x


class DecoderResidualBlock(nn.Module):

    def __init__(self, hidden_channels, output_channels, layers):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels,
                                             upsample=True)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels,
                                             upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderBottleneckBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, layers):
        super(DecoderBottleneckBlock, self).__init__()

        for i in range(layers):

            if i == layers - 1:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=down_channels, upsample=True)
            else:
                layer = DecoderBottleneckLayer(in_channels=in_channels, hidden_channels=hidden_channels,
                                               down_channels=in_channels, upsample=False)

            self.add_module('%02d EncoderLayer' % i, layer)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x


class DecoderResidualLayer(nn.Module):

    def __init__(self, hidden_channels, output_channels, upsample):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False)
            )
        else:
            self.weight_layer2 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1,
                          bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        else:
            self.upsample = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)

        if self.upsample is not None:
            identity = self.upsample(identity)

        x = x + identity

        return x


class DecoderBottleneckLayer(nn.Module):

    def __init__(self, in_channels, hidden_channels, down_channels, upsample):
        super(DecoderBottleneckLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
        )

        self.weight_layer2 = nn.Sequential(
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )

        if upsample:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        else:
            self.weight_layer3 = nn.Sequential(
                nn.BatchNorm2d(num_features=hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0,
                          bias=False)
            )

        if upsample:
            self.upsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=2,
                                   output_padding=1, bias=False)
            )
        elif (in_channels != down_channels):
            self.upsample = None
            self.down_scale = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channels, out_channels=down_channels, kernel_size=1, stride=1, padding=0,
                          bias=False)
            )
        else:
            self.upsample = None
            self.down_scale = None

    def forward(self, x):

        identity = x

        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        x = self.weight_layer3(x)

        if self.upsample is not None:
            identity = self.upsample(identity)
        elif self.down_scale is not None:
            identity = self.down_scale(identity)

        x = x + identity

        return x

