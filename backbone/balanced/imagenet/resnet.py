import torch
import timm
from torch import nn
import torch.nn.functional as F

class resnet50(nn.Module):
    def __init__(self, output_dim=1000, model_name="resnet50"):
        super(resnet50, self).__init__()
        self.backbone = timm.create_model(model_name=model_name, features_only=True, pretrained=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim, bias=False)
        self.BN_H = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x)
        return x