import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,Subset
from torch.nn import CrossEntropyLoss,TripletMarginLoss
from torch.optim import Adam
from torch.optim import lr_scheduler
from Model import Train_Model
from cross_entropy_smooth import CrossEntropySmooth
import os
import numpy as np
import time

os.chdir("D:\\Git_Repo\\Tricks Rebuild\\")
model = Train_Model()

#Warmup Learning Rate
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
#Get Each Classes Num
def Classes_Num(root):
    dirs = os.listdir(root)
    classes_num = len(dirs)
    class_num = torch.zeros((classes_num))
    for i,sub_dir in enumerate(dirs):
        class_num[i] = len(os.listdir(root+sub_dir))
    return class_num

#Select Hard-Triplets
def Select_Triplet(embdings,index,B = 16,P = 4):
    triplets = []
    embdings = torch.zeros_like(embdings).scatter_(0,torch.tensor(index,dtype = torch.long),embdings)
    for i in range(0,B):
        anchor_emb = embdings[i*4:i*4+4].cpu().detach().numpy()
        for j in range(0,P):
            temp_anchor = anchor_emb[j]
            temp_positive = 0.0
            temp_positive_index = 0
            temp_negtive = 99999
            temp_negtive_index = 0
            for m in range(0,P):
                if m == j:
                    continue
                pos_distance = np.sum(np.square(temp_anchor - anchor_emb[m]))
                if pos_distance > temp_positive:
                    temp_positive == pos_distance
                    temp_positive_index = m
                for k,emb in enumerate(embdings):
                    if k <i*4+4 and k >= i*4:
                        continue
                    neg_emb = emb.cpu().detach().numpy()
                    #TODO
                    neg_distance = np.sum(np.square(temp_anchor-neg_emb))
                    #neg_index = np.argsort(neg_distance)[0]
                    #neg_distance = neg_distance[neg_index]
                    if neg_distance < temp_negtive:
                        temp_negtive = neg_distance
                        temp_negtive_index = k
            triplets.append({'a':i*4+j,'p':i*4+temp_positive_index,'n':temp_negtive_index})
    return triplets







#dataset&loader
train_root = './Market/pytorch/train_all'
train_dataset = ImageFolder(train_root,transform=train_transform)
val_root = './Market/pytorch/val'
val_dataset = ImageFolder(val_root,transform=val_transorm)
#trainloader = DataLoader(train_dataset,64)
valloader = DataLoader(val_dataset,64,True)

#EPOCH
EPOCH = 120

#If support GPU
if torch.cuda.is_available():
    print("Support GPU!")
    device = torch.device('cuda:4')
else:
    device = torch.device('cpu')
model = model.to(device)
#Total Class
train_class = Classes_Num(train_root + '/')
val_class = Classes_Num(val_root + '/')
#Set optimizer
optimizer = Adam(model.parameters(),lr = 3.5e-5)
#Set warmup strategy
warmup = lr_scheduler.LambdaLR(optimizer,WarmUp_Scheduler)
#Label Smooth Cross Entropy
IDloss = CrossEntropySmooth(classes_num = 751)
#Triplet Loss
triloss = TripletMarginLoss(margin=0.3)
#Batch Size
batch_size = 64
total_train = int(torch.sum(train_class)/batch_size)
total_val = int(torch.sum(val_class)/batch_size)
B = 16
P = 4

Pretrain = False
path_to_model = ''
if Pretrain == True:
    model.load_state_dict(torch.load(path_to_model))
#Training
for epoch in range(EPOCH):
    print('Epoch {}/{}'.format(epoch, EPOCH - 1))
    print('-' * 10)
    warmup.step()
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    start = time.clock()
    for phase in ['Train','Val']:
        if phase == 'Train':
            for i in range(total_train):
                embdings = []
                Inputs,Labels,index_ = RandomPic(classes_num = train_class,dataset=train_dataset,B = B,P=P)
                optimizer.zero_grad()
                corrects = 0.0
                Inputs = Inputs.to(device)
                Labels = Labels.to(device)
                ft,fi,out = model(Inputs)
                _,preds = torch.max(out,1)
                corrects = torch.sum(preds == Labels)
                running_corrects += float(corrects)
                idloss = IDloss(out,Labels)
                triplets_index = Select_Triplet(ft,index_)
                #Calculate Triplet Loss
                num_of_triplets = len(triplets_index)
                #triplet_a = torch.unsqueeze(ft[np.argwhere(index_ == triplets_index[0]['a'])[0,0]],dim = 0)
                #triplet_p = torch.unsqueeze(ft[np.argwhere(index_ == triplets_index[0]['p'])[0,0]],dim = 0)
                #triplet_n = torch.unsqueeze(ft[np.argwhere(index_ == triplets_index[0]['n'])[0,0]],dim = 0)
                triplet_loss = 0.0
                for i in range(0,num_of_triplets):
                    triplet_a = torch.unsqueeze(ft[np.argwhere(index_ == triplets_index[i]['a'])[0,0]],dim = 0)
                    triplet_p = torch.unsqueeze(ft[np.argwhere(index_ == triplets_index[i]['p'])[0,0]],dim = 0)
                    triplet_n = torch.unsqueeze(ft[np.argwhere(index_ == triplets_index[i]['n'])[0,0]],dim = 0)
                    triplet_loss += triloss(triplet_a,triplet_p,triplet_n)
                triplet_loss /= num_of_triplets
                total_loss = idloss+triplet_loss
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss
            running_loss /= total_train
            running_corrects /= (total_train * 64)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase,running_loss, running_corrects))
            duration = (time.clock() - start)
            print('Training complete in {:.0f}m {:.0f}s'.format(
                    duration // 60, duration % 60))
            print()
            running_loss = 0.0
            running_corrects = 0.0
        if phase == 'Val':
            turn = 0.0
            for index,data in enumerate(valloader):
                turn += 1
                Inputs,Labels = data
                Inputs = Inputs.to(device)
                Labels = Labels.to(device)
                with torch.no_grad():
                    ft,fi,out = model(Inputs)
                    idloss = IDloss(out,Labels)
                    _,preds = torch.max(out,1)
                    corrects = torch.sum(preds == Labels)
                    running_corrects += float(corrects)
                    running_loss += idloss
            running_loss /= turn
            running_corrects /= (turn * 64)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase,running_loss, running_corrects))
            print()
    if epoch % 10 == 9:
        torch.save(model.state_dict(),'./Model0_' + str(epoch) + '.pth')
