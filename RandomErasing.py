from torchvision.transforms import *
import random
import math
import torch


class RandomErasing(object):
    def __init__(self, prob=0.5, sel=0.002, seh=0.4, r1=0.3, r2=3.33):
        super(RandomErasing, self).__init__()
        self.prob = prob
        self.sel = sel
        self.seh = seh
        self.r1 = r1
        self.r2 = r2

    def __call__(self, img):
        if random.uniform(0, 1) > self.prob:
            return img
        for i in range(100):
            area = img.size()[1]*img.size()[2]
            target_area = random.uniform(self.sel, self.seh)*area
            aspect_ratio = random.uniform(self.r1, self.r2)
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            h = int(round(math.sqrt(target_area*aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                y = random.randint(0, img.size()[1] - h)
                x = random.randint(0, img.size()[2] - w)
                img[0, y:y+h, x:x+w] = 0.485
                img[1, y:y+h, x:x+w] = 0.456
                img[2, y:y+h, x:x+w] = 0.406
                return img
        return img
