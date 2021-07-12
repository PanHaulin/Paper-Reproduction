import torch.nn as nn
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# 有一定问题，plain无法收敛，参考Res2Net

def conv3x3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, plain=False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # self.downsample = downsample
        self.plain = plain
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)


        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # if self.plain == False:
        #     out = out + residual

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, plain=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.plain= plain

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
        
        if self.plain == False:
            print('residual')
            out = out + residual

        out = self.relu(out)

        return out

class SimpleResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, plain=False):
        super().__init__()
        self.inplanes = 32
        self.plain = plain
        self.num_classes = num_classes
        self.num_blocks = num_blocks

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.linear = nn.Linear(64, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
 
        self.fully_conv = False
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0,0.01) # 遵循CNN中的，零均值高斯分布初始化，标准偏差为1
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                m.weight.data.normal_(0,0.01) # 没提到，因此遵循CNN中的权重初始化
                m.bias.data.zero_() #用恒均值0初始化剩余层偏差
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if self.plain == False and (stride !=1 or self.inplanes != planes * block.expansion):
        #     # 如果特征图size不同 或 通道数不同
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes * block.expansion)
        #     )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, plain=self.plain)) # 有下采样的层
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, plain=self.plain)) # stride=1, downsample=None
        
        return nn.Sequential(*layers)
 
    def forward(self, x):

        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)

        if not self.fully_conv:
            x = torch.flatten(x, 1) # 拉平

        x = self.fc(x)

        if self.fully_conv:
            x = x.view(x.size(0), -1)

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, plain=False):
        super().__init__()
        self.inplanes = 64
        self.plain = plain
        self.num_classes = num_classes
        self.layers = layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # size变化
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.fully_conv = False
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0,0.01) # 遵循CNN中的，零均值高斯分布初始化，标准偏差为1
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                m.weight.data.normal_(0,0.01) # 没提到，因此遵循CNN中的权重初始化
                m.bias.data.zero_() #用恒均值0初始化剩余层偏差
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if self.plain == False and (stride !=1 or self.inplanes != planes * block.expansion):
            # 如果特征图size不同 或 通道数不同
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, plain=self.plain)) # 有下采样的层
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, plain=self.plain)) # stride=1, downsample=None
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        if not self.fully_conv:
            x = torch.flatten(x, 1) # 拉平

        x = self.fc(x)

        if self.fully_conv:
            x = x.view(x.size(0), -1)

        return x

def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet20(**kwargs):
    model = SimpleResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32(**kwargs):
    model = SimpleResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet44(**kwargs):
    model = SimpleResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet56(**kwargs):
    model = SimpleResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet110(**kwargs):
    model = SimpleResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet1202(**kwargs):
    model = SimpleResNet(BasicBlock, [200, 200, 200], **kwargs)
    return model

def create_model(version, num_classes, plain):
    if version == 18:
        model = resnet18(num_classes=num_classes, plain=plain)
    elif version == 20:
        model = resnet20(num_classes=num_classes, plain=plain)
    elif version == 32:
        model = resnet32(num_classes=num_classes, plain=plain)
    elif version == 34:
        model = resnet34(num_classes=num_classes, plain=plain)
    elif version == 44:
        model = resnet44(num_classes=num_classes, plain=plain)
    elif version == 50:
        model = resnet50(num_classes=num_classes, plain=plain)
    elif version == 56:
        model = resnet56(num_classes=num_classes, plain=plain)
    elif version == 101:
        model = resnet101(num_classes=num_classes, plain=plain)
    elif version == 110:
        model = resnet110(num_classes=num_classes, plain=plain)
    elif version == 152:
        model = resnet152(num_classes=num_classes, plain=plain)
    elif version == 1202:
        model = resnet1202(num_classes=num_classes, plain=plain)
    else:
        print('unknown version. must be 18/20/32/34/44/50/56/101/110/152/1202')
        return
    return model