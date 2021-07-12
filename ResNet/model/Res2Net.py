from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch.nn as nn
import torch

# 基于Pytorch实现ResNet论文复现

class PlainBasicBlock(BasicBlock):
    # def __init__(self,
    #             inplanes: int, 
    #             planes: int,
    #             stride: int = 1,
    #             downsample: Optional[nn.Module] = None,
    #             groups: int = 1,
    #             base_width: int = 64,
    #             dilation: int = 1,
    #             norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
    #     super().__init__(inplanes=inplanes, planes=planes, stride=stride,downsample=downsample, groups=groups, base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    
    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)
        return out

class PlainBottleneck(Bottleneck):
    # def __init__(self,
    #             inplanes: int,
    #             planes: int,
    #             stride: int = 1,
    #             downsample: Optional[nn.Module] = None,
    #             groups: int = 1,
    #             base_width: int = 64,
    #             dilation: int = 1,
    #             norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
    #     super().__init__(inplanes=inplanes, planes=planes, stride=stride,downsample=downsample, groups=groups, base_width=base_width, dilation=dilation, norm_layer=norm_layer)
    
    def forward(self, x: Tensor) -> Tensor:

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)
        return out

class ZeroPad(nn.Module):
    def __init__(self, inplane, outplane):
        super(ZeroPad, self).__init__()
        self.addplane = outplane-inplane
    
    
    def forward(self, x):
        y = torch.zeros([x.size(0), self.addplane, x.size(2), x.size(3)]).type_as(x)
        # print(x.size())
        # print(y.size())
        x = torch.cat((x,y), dim=1)
        return x

class Res2Net(ResNet):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int, option='B') -> None:
        self.option=option
        super().__init__(block, layers, num_classes=num_classes)
        self.fully_conv =  False
        print('Use Option {}'.format(option))
    
    def _forward_impl(self, x: Tensor) -> Tensor:

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
            x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.fully_conv:
            x = x.view(x.size(0), -1)

        return x
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        

        if self.option == 'A':
            if self.inplanes != planes * block.expansion:
                if stride != 1:
                    downsample = nn.Sequential(
                        ZeroPad(self.inplanes, planes * block.expansion),
                        conv1x1(planes * block.expansion, planes * block.expansion, stride),
                        norm_layer(planes * block.expansion),
                    )
                else:
                    downsample = nn.Sequential(
                        ZeroPad(self.inplanes, planes * block.expansion),
                        norm_layer(planes * block.expansion),
                    )
        elif self.option == 'B':
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
        else:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> Res2Net:
    model = Res2Net(block, layers, **kwargs)
    return model


def resnet18(plain=False, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Res2Net:
    if not plain:
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)
    else:
        return _resnet('resnet18', PlainBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(plain=False, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Res2Net:
    if not plain:
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)
    else:
        return _resnet('resnet34', PlainBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(plain=False, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Res2Net:
    if not plain:
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,**kwargs)
    else:
        return _resnet('resnet50', PlainBottleneck, [3, 4, 6, 3], pretrained, progress,**kwargs)



def resnet101(plain=False, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Res2Net:
    if not plain:
        return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
    else:
        return _resnet('resnet101', PlainBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(plain=False, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Res2Net:
    if not plain:
        return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)
    else:
        return _resnet('resnet152', PlainBottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)

def create_model(version, num_classes, plain, option='B'):
    if version == 18:
        model = resnet18(num_classes=num_classes, plain=plain)
    elif version == 34:
        model = resnet34(num_classes=num_classes, plain=plain, option=option)
    elif version == 50:
        model = resnet50(num_classes=num_classes, plain=plain)
    elif version == 101:
        model = resnet101(num_classes=num_classes, plain=plain)
    elif version == 152:
        model = resnet152(num_classes=num_classes, plain=plain)
    else:
        print('unknown version. must be 18/20/32/34/44/50/56/101/110/152/1202')
        return
    return model