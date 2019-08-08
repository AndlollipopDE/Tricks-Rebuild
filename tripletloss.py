import torch
from torch.nn import TripletMarginLoss

def Distance(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def Hard_triplet(embdings,num_instance):
    dist_map = Distance(embdings,embdings)
    N = embdings.size(0)
    num_person = N // num_instance
    triplets = []
    for i in range(num_person):
        start = i * num_instance
        for j in range(num_instance):
            start_ = start + j
            person_dist = dist_map[start_]
            max_idx = torch.max(person_dist.detach()[start:start+num_instance],dim = 0)[1].item() + start
            tmp_dist = person_dist.detach()
            tmp_dist[start:start+num_instance] = 99999
            min_idx = torch.min(tmp_dist,dim = 0)[1].item()
            triplets.append([start_,max_idx,min_idx])
    return triplets

def TripletLoss(embdings,margins,num_instance):
    triplets = Hard_triplet(embdings.detach(),num_instance)
    loss = 0.0
    tripletloss = TripletMarginLoss(margin=margins)
    for triplet in triplets:
        a = triplet[0]
        p = triplet[1]
        n = triplet[2]
        loss += tripletloss(torch.unsqueeze(embdings[a]),torch.unsqueeze(embdings[p]),torch.unsqueeze(embdings[n]))
    return loss


