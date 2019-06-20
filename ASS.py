import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from torch.nn import CrossEntropyLoss,TripletMarginLoss
from torch.optim import Adam
from torch.optim import lr_scheduler
from Model import Train_Model
from RandomErasing import RandomErasing
from cross_entropy_smooth import CrossEntropySmooth
import os
import numpy as np
import time

def WarmUp_Scheduler(epoch):
    if epoch <= 10:
        return epoch
    elif epoch <= 40 and epoch > 10:
        return 1
    elif epoch == 41:
        return 0.1
    elif epoch > 41 and epoch <= 70:
        return 1
    elif epoch == 71:
        return 0.1
    elif epoch > 71:
        return 1
model = Train_Model()
if torch.cuda.is_available():
    print("Support GPU!")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = model.to(device)
optimizer = Adam(model.parameters(),lr = 3.5e-5)
warmup = lr_scheduler.LambdaLR(optimizer,WarmUp_Scheduler)
for i in range(0,100):
    warmup.step()
    print(optimizer)
    