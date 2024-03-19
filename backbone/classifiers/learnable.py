from __future__ import absolute_import
import torch.nn as nn
from torch.nn.functional import linear, normalize

class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_in, num_classes,bias=False)

    def forward(self, x):
        x = normalize(x) # Normalize the input feature
        weight = normalize(self.fc.weight)  # Normalize the weight tensor
        logits = linear(x, weight)

        return logits