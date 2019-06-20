import torch
import numpy as np
import time

def VS():
    randn1 = torch.randn(2048)
    randn2 = torch.randn(64,2048)
    randn3 = randn1.numpy()
    randn4 = randn2.numpy()
    start = time.clock()
    for i in range(0,100):
        distance = np.sqrt(np.sum(np.square(randn3 - randn4),1))
        haha = np.min(distance)
    duration = (time.clock() - start)
    print(duration)
    print(haha)
    distance = torch.zeros(64)
    device = torch.device('cuda')
    randn1 = randn1.to(device)
    randn2 = randn2.to(device)
    start = time.clock()
    for k in range(0,100):
        for i in range(0,64):
            distance[i] = torch.dist(randn1,randn2[i])
        haha = torch.min(distance)
    duration = (time.clock() - start)
    print(duration)
    print(haha)


VS()
    