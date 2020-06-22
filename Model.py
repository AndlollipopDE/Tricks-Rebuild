from torchvision.models import resnet50
import torch
import torch.nn as nn
from torch.nn import Module

# initialize


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        #nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num,  relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        #add_block = []
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        #add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block1(x)
        x1 = self.add_block2(x)
        x2 = self.classifier(x1)
        return x, x1, x2

# model


class Train_Model(Module):
    def __init__(self):
        super(Train_Model, self).__init__()
        #self.part = 6
        self.baseline = resnet50(pretrained=True)
        #self.dropout = nn.Dropout(0.5)
        # Last Stride
        self.baseline.layer4[0].conv2.stride = (1, 1)
        self.baseline.layer4[0].downsample[0].stride = (1, 1)
        self.baseline.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.baseline.fc = nn.Linear(2048, 751, False)
        self.baseline.fc.apply(weights_init_classifier)
        self.BN = nn.BatchNorm1d(2048)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(weights_init_kaiming)

    def forward(self, x):
        # baseline
        x = self.baseline.conv1(x)
        x = self.baseline.bn1(x)
        x = self.baseline.relu(x)
        x = self.baseline.maxpool(x)
        x = self.baseline.layer1(x)
        x = self.baseline.layer2(x)
        x = self.baseline.layer3(x)
        x = self.baseline.layer4(x)
        # BNNeck
        ft = self.baseline.avgpool(x).squeeze()
        fi = self.BN(ft)
        out = self.baseline.fc(fi)
        return ft, fi, out


class Test_Model(Module):
    def __init__(self, model):
        super(Test_Model, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model.baseline.conv1(x)
        x = self.model.baseline.bn1(x)
        x = self.model.baseline.relu(x)
        x = self.model.baseline.maxpool(x)
        x = self.model.baseline.layer1(x)
        x = self.model.baseline.layer2(x)
        x = self.model.baseline.layer3(x)
        x = self.model.baseline.layer4(x)
        # BNNeck
        ft = self.model.baseline.avgpool(x).squeeze()
        fi = self.model.BN(ft)
        return fi
