'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, 
                conv_layer=None,
                norm_layer=None,
                activation_layer=None):
        super(BasicBlock, self).__init__()
        print(activation_layer)
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(planes)
        self.relu = activation_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,
                conv_layer=None,
                norm_layer=None,
                activation_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = norm_layer(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.norm2 = norm_layer(planes)
        self.conv3 = conv_layer(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.norm3 = norm_layer(self.expansion*planes)
        self.relu = activation_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv_layer(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(BaseModel):
    def __init__(self, block, num_blocks, num_classes=10,
                norm_layer_type = 'bn',
                conv_layer_type = 'conv2d',
                linear_layer_type = 'linear',
                activation_layer_type = 'relu',
                etf_fc = False):
        super().__init__(norm_layer_type, conv_layer_type, linear_layer_type, 
                        activation_layer_type)
        
        self.in_planes = 64

        self.conv1 = self.conv_layer(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm1 = self.norm_layer(64)
        self.relu = self.activation_layer(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = self.linear_layer(512*block.expansion, 360)
#         self.linear2 = self.linear_layer(360, num_classes)
        self.linear = self.linear_layer(512*block.expansion, num_classes) #, bias=False
        
        ################ #Added for ETF classifier #################
        ############################################################
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if etf_fc:
                    print("Use ETF classifier")
                    weight = torch.sqrt(torch.tensor(num_classes/(num_classes-1)))*(torch.eye(num_classes)-(1/num_classes)*torch.ones((num_classes, num_classes)))
                    weight /= torch.sqrt((1/num_classes*torch.norm(weight, 'fro')**2))
                    if False: #fixdim
                        m.weight = nn.Parameter(weight)
                    else:
                        m.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, 512 * block.expansion)))
                    m.weight.requires_grad_(False)
        ############################################################
        ############################################################
                    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                                self.conv_layer, self.norm_layer, 
                                self.activation_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        logits = self.linear(features)
#         out = self.linear2(out) # added
        return logits
        #return features, logits


def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)