import numpy as np
import torch


def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
#        length = (gamma / label_count)
        #length = torch.sqrt(label_count / gamma)
        Ew[0, label_id] = length
    return Ew

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def LTloss(feat,  target, reg_lam=0):

    with torch.no_grad():
        feat_nograd = feat.detach()
        H_length = torch.clamp(torch.sqrt(torch.sum(feat_nograd ** 2, dim=1, keepdims=False)), 1e-8)
        M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
    dot =  torch.bmm(feat.unsqueeze(1), target.unsqueeze(2)).view(-1)
    loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

    if reg_lam > 0:
        reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(feat ** 2, dim=1, keepdims=True)))
        loss = loss + reg_Eh_l2*reg_lam

    return loss