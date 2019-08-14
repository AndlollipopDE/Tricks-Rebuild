from torch.optim import Adam
from torch.optim import lr_scheduler
import torch
def WarmUp_Scheduler(epoch):
    if epoch <= 10:
        return epoch * 10
    elif epoch <= 60 and epoch > 10:
        return 100
    elif epoch > 60 and epoch <= 130:
        return 10
    elif epoch > 130 and epoch <= 300:
        return 1
    elif epoch > 300:
        return 0.1

def Make_Optimizer(model,learning_rate):
    '''
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = learning_rate
        weight_decay = 0.0005
        if "bias" in key:
            lr = learning_rate
            weight_decay = 0.0005
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = Adam(params)
    '''

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate,weight_decay=5e-4,momentum=0.9, nesterov=True)
    return optimizer

def Warmup(optimizer):
    warmup = lr_scheduler.LambdaLR(optimizer,WarmUp_Scheduler)
    return warmup