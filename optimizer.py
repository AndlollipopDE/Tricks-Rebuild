from torch.optim import Adam
from torch.optim import lr_scheduler

def WarmUp_Scheduler(epoch):
    if epoch <= 10:
        return epoch
    elif epoch <= 40 and epoch > 10:
        return 10
    elif epoch == 41:
        return 0.1
    elif epoch > 41 and epoch <= 70:
        return 0.1
    elif epoch == 71:
        return 0.01
    elif epoch > 71:
        return 0.01

def Make_Optimizer(model,learning_rate):
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
    return optimizer

def Warmup(optimizer):
    warmup = lr_scheduler.LambdaLR(optimizer,WarmUp_Scheduler)
    return warmup