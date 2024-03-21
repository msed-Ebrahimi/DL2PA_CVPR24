import numpy as np
import torch

def BLoss(criterion, feat, targetP):
    return (1.0 - criterion(feat, targetP)).pow(2).sum()
