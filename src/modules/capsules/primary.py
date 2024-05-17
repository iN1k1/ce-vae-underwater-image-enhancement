import torch
import torch.nn as nn
from src.modules.capsules.common import squash


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules: int = 16):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=2, padding=2)
            for _ in range(num_capsules)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.permute(0, 2, 3, 4, 1)
        u = u.view(x.shape[0], u.shape[1] * u.shape[2] * u.shape[3], -1)
        return squash(u)
