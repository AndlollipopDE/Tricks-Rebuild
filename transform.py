import torchvision.transforms as transforms
from randomerasing import RandomErasing

def Train_Transform(RandomER = False):
    train_transform_list = [transforms.Resize((256,128)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.Pad(10),
                    transforms.RandomCrop((256,128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])]
    if RandomER:
        REA = RandomErasing()
        train_transform_list.append(REA)
    train_transform = transforms.Compose(train_transform_list)
    return train_transform


def Val_Transform():
    val_tansform_list = [transforms.Resize((256,128)),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])]
    val_transorm = transforms.Compose(val_tansform_list)
    return val_transorm
    