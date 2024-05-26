import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def soft_process(loss):
    soft_loss = torch.log(1+loss+loss*loss/2)
    return soft_loss

def PLC_uncertain_discard(user, item, train_mat, y, t, drop_rate, epoch, sn, before_loss, co_lambda, relabel_ratio):
    # This is the core of our DCF
    before_loss = torch.from_numpy(before_loss).cuda().float().squeeze()
    
    s = torch.tensor(epoch + 1).float() # as the epoch starts from 0
    co_lambda = torch.tensor(co_lambda).float()    
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)
    loss_mul = loss * t
    loss_mul = soft_process(loss_mul)  # soft process is non-decreasing damping function in the paper
    
    loss_mean = (before_loss * s + loss_mul) / (s + 1)   # computing mean loss in Eq.2
    confidence_bound = co_lambda * (s + (co_lambda * torch.log(2 * s)) / (s * s)) / ((sn + 1) - co_lambda)
    confidence_bound = confidence_bound.squeeze()
    loss_mul = F.relu(loss_mean.float() - confidence_bound.cuda().float())  # loss low bound in Eq.4
    
    # The following is the code implementation of dropping and relabelling.
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]
    
    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))
    
    highest_ind_sorted = ind_sorted[int(((1-relabel_ratio)+relabel_ratio*remember_rate)*len(loss_sorted)):]
    saved_ind_sorted = ind_sorted[:num_remember]
    final_ind = torch.concat((highest_ind_sorted, saved_ind_sorted))
    lowest_ind_sorted = ind_sorted[:int(((1-relabel_ratio)+relabel_ratio*remember_rate)*len(loss_sorted))]

    t = torch.tensor(train_mat[user.cpu().numpy().tolist(), item.cpu().numpy().tolist()].todense()).squeeze().cuda()
    loss_update = F.binary_cross_entropy_with_logits(y[final_ind], t[final_ind])
    
    return loss_update, train_mat, loss_mean, lowest_ind_sorted    


def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update
