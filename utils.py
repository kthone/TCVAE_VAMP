from numbers import Number
import math
import torch
import os


def save_checkpoint(state, save, epoch, idx):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-{}-{}.pth'.format(epoch, idx))
    torch.save(state, filename)
