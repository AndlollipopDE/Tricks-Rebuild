from torchvision.models import resnet50
import torch
import torch.nn as nn
from torch.nn import Module

#initialize
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
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

#model
class Train_Model(Module):
    def __init__(self):
        super(Train_Model,self).__init__()
        self.baseline = resnet50(pretrained=True)
        #Last Stride
        self.baseline.layer4[0].conv2.stride = (1,1)
        self.baseline.layer4[0].downsample[0].stride = (1,1)
        self.baseline.fc = nn.Linear(2048,751,False)
        self.BN = nn.BatchNorm1d(2048)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(weights_init_kaiming)
        self.baseline.fc.apply(weights_init_classifier)
    def forward(self,x):
        #baseline
        x = self.baseline.conv1(x)
        x = self.baseline.bn1(x)
        x = self.baseline.relu(x)
        x = self.baseline.maxpool(x)
        x = self.baseline.layer1(x)
        x = self.baseline.layer2(x)
        x = self.baseline.layer3(x)
        x = self.baseline.layer4(x)
        #BNNeck
        ft = self.baseline.avgpool(x).squeeze()
        fi = self.BN(ft)
        out = self.baseline.fc(fi)
        return ft,fi,out

class Test_Model(Module):
    def __init__(self,model):
        super(Test_Model,self).__init__()
        self.model = model
    def forward(self,x):
        x = self.model.baseline.conv1(x)
        x = self.model.baseline.bn1(x)
        x = self.model.baseline.relu(x)
        x = self.model.baseline.maxpool(x)
        x = self.model.baseline.layer1(x)
        x = self.model.baseline.layer2(x)
        x = self.model.baseline.layer3(x)
        x = self.model.baseline.layer4(x)
        #BNNeck
        ft = self.model.baseline.avgpool(x).squeeze()
        fi = self.model.BN(ft)
        return fi

