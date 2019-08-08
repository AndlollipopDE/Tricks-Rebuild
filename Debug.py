import torch
from dataset import ImageDataset, RandomIdentitySampler
from market1501 import Market1501
from transform import Train_Transform, Val_Transform
from torch.utils.data import DataLoader
from model import Train_Model
from tripletloss import TripletLoss
from cross_entropy_smooth import CrossEntropySmooth
from optimizer import Make_Optimizer, Warmup
import os
import time

torch.cuda.current_device()
torch.cuda._initialized = True
if torch.cuda.is_available():
    print("Support GPU!")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
os.chdir('D:\\Git_Repo\\Market\\')
market = Market1501(root='./')
train_transform = Train_Transform(True)
val_transform = Val_Transform()
train_sampler = RandomIdentitySampler(market.train, 16, 4)
train_dataset = ImageDataset(dataset=market.train, transform=train_transform)
val_dataset = ImageDataset(dataset = market.test + market.query, transform = val_transform)
train_dataloader = DataLoader(train_dataset, 64, False, train_sampler)
val_dataloader = DataLoader(val_dataset,128,False)
Model = Train_Model().to(device)
IDloss = CrossEntropySmooth(market.num_train_id)
optimizer = Make_Optimizer(Model, 3.5e-5)
warmup = Warmup(optimizer)
EPOCH = 120
for epoch in range(EPOCH):
    print('Epoch {}/{}'.format(epoch, EPOCH - 1))
    print('-' * 10)
    warmup.step()
    Model.train()
    for phase in ['Train', 'Val']:
        if phase == 'Train':
            start = time.clock()
            running_loss = 0.0
            running_corrects = 0.0
            running_times = 0.0
            for index, data in enumerate(train_dataloader):
                running_times += 1
                Data, Label = data
                Data = Data.to(device)
                Label = Label.to(device)
                ft, fi, out = Model(Data)
                _, preds = torch.max(out,1)
                corrects = torch.sum(preds == Label)
                running_corrects += float(corrects)
                idloss = IDloss(out, Label)
                triloss = TripletLoss(ft, 0.3, 4)
                loss = idloss+triloss
                running_loss += loss
                loss.backward()
                optimizer.step()
            running_loss /= running_times
            running_corrects /= market.num_train_img
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase,running_loss, running_corrects))
            duration = (time.clock() - start)
            print('Training complete in {:.0f}m {:.0f}s'.format(
                    duration // 60, duration % 60))
        if phase == 'Val':
            start = time.clock()
            running_loss = 0.0
            running_corrects = 0.0
            running_times = 0.0
            for index,data in enumerate(val_dataloader):
                running_times += 1
                Data, Label = data
                Data = Data.to(device)
                Label = Label.to(device)
                with torch.no_grad():
                    ft,fi,out = Model(Data)
                    idloss = IDloss(out,Label)
                    triloss = TripletLoss(ft,0.3,4)
                    _,preds = torch.max(out,1)
                    corrects = torch.sum(preds == Label)
                    running_corrects += float(corrects)
                    running_loss += (idloss+triloss)
            running_loss /= running_times
            running_corrects /= (market.num_query_img + market.num_test_img)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase,running_loss, running_corrects))
            duration = (time.clock() - start)
            print('Val complete in {:.0f}m {:.0f}s'.format(
                    duration // 60, duration % 60))
            print()
			
    if epoch % 10 == 9:
        torch.save(Model.state_dict(),'./Model0_' + str(epoch) + '.pth')