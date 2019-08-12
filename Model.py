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
        self.part = 6
        self.baseline = resnet50(pretrained=True)
        self.dropout = nn.Dropout(0.5)
        #Last Stride
        self.baseline.layer4[0].conv2.stride = (1,1)
        self.baseline.layer4[0].downsample[0].stride = (1,1)
        #self.baseline.fc = nn.Linear(2048,751,False)
        #self.BN = nn.BatchNorm1d(2048)
        #self.BN.bias.requires_grad_(False)
        #self.BN.apply(weights_init_kaiming)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        for i in range(self.part):
            bn = nn.BatchNorm1d(2048)
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
            setattr(self,'bn_1_' + str(i + 1),bn)
            fc = nn.Linear(2048,256,False)
            fc.apply(weights_init_classifier)
            setattr(self,'fc' + str(i + 1),fc)
            bn = nn.BatchNorm1d(256)
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)
            setattr(self,'bn_2_' + str(i + 1),bn)
            setattr(self,'dropout' + str(i + 1),nn.Dropout(0.5))
            classifier = nn.Linear(256,751,False)
            classifier.apply(weights_init_classifier)
            setattr(self,'classifier' + str(i + 1),classifier)
            
        #self.baseline.fc.apply(weights_init_classifier)
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
        #ft = self.baseline.avgpool(x).squeeze()
        #fi = self.BN(ft)
        #out = self.baseline.fc(fi)
        x = self.avgpool(x)
        x = self.dropout(x)
        parts = []
        predict = []
        fts = []
        fi = None
        for i in range(self.part):
            parts.append(torch.squeeze(x[:,:,i]))
        for i in range(self.part):
            bn = getattr(self,'bn_1_' + str(i + 1))
            x = bn(parts[i])
            fc = getattr(self,'fc' + str(i + 1))
            x = fc(x)
            bn = getattr(self,'bn_2_' + str(i + 1))
            x = bn(x)
            dropout = getattr(self,'dropout' + str(i + 1))
            x = dropout(x)
            classifier = getattr(self,'classifier' + str(i + 1))
            x = classifier(x)
            predict.append(x)
        ft = torch.cat(parts,1)
        return ft,fi,predict

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
        #ft = self.model.baseline.avgpool(x).squeeze()
        #fi = self.model.BN(ft)
        x = self.model.avgpool(x)
        x = self.model.dropout(x)
        parts = []
        fis = []
        for i in range(self.model.part):
            parts.append(torch.squeeze(x[:,:,i]))
        for i in range(self.model.part):
            bn = getattr(self.model,'bn_1_' + str(i + 1))
            x = bn(parts[i])
            fis.append(x)
        fi = torch.cat(fis,1)
        return fi

