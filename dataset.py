from torch.utils.data import Dataset,Sampler
from PIL import Image
from collections import defaultdict
import copy
import numpy
import random

class ImageDataset(Dataset):
    def __init__(self,dataset,transform = None):
        super(ImageDataset,self).__init__()
        self.dataset = dataset
        self.transform = transform
    def read_image(self,img_path):
        img = Image.open(img_path).convert('RGB')
        return img
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        img_path,personid = self.dataset[index]
        img = self.read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img,personid

class RandomIdentitySampler(Sampler):
    def __init__(self,data_source,batch_size,num_instance):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.num_personid = self.batch_size // self.num_instance
        self.index_dict = defaultdict(list)
        for index,filename in enumerate(self.data_source):
            personid = filename[1]
            self.index_dict[personid].append(index)
        self.personids = list(self.index_dict.keys())
        self.length = 0
        for personid in self.personids:
            idx = self.index_dict[personid]
            num = len(idx)
            if num < self.num_instance:
                num = self.num_instance
            self.length += num - num % self.num_instance
    def __iter__(self):
        batch_index_dict = defaultdict(list)
        for personid in self.personids:
            indexs = copy.deepcopy(self.index_dict[personid])
            if len(indexs) < self.num_instance:
                indexs = numpy.random.choice(indexs,size = self.num_instance,replace = True)
            batch_index = []
            for index in indexs:
                batch_index.append(index)
                if len(batch_index) == self.num_instance:
                    batch_index_dict[personid].append(batch_index)
                    batch_index = []
        personids = copy.deepcopy(self.personids)
        final_indexs = []
        while len(personids) > self.num_personid:
            id_choices = random.sample(personids,self.num_personid)
            for ids in id_choices:
                person_id = batch_index_dict[ids].pop(0)
                if len(batch_index_dict[ids]) == 0:
                    personids.remove(ids)
                final_indexs.extend(person_id)
        self.length = len(final_indexs)
        return iter(final_indexs)
    def __len__(self):
        return self.length