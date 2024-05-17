from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from src.modules.capsules.common import squash
import torch.nn.functional as F


class DigitCaps(nn.Module):
    def __init__(self, channels: int, logits_num: int = 64, num_routes: Tuple[int] = (32, 9, 9),
                 num_capsules: int = 16):
        super(DigitCaps, self).__init__()
        self.num_routes = num_routes
        self.W = nn.Parameter(torch.randn(1, np.prod(self.num_routes), 1, logits_num, num_capsules))
        self.out_conv = nn.ConvTranspose2d(num_routes[0], channels, kernel_size=2, stride=2,
                                           padding=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * 1, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        num_routes = np.prod(self.num_routes)
        b_ij = torch.zeros(1, num_routes, 1, 1)
        b_ij = b_ij.to(x.device)

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
            else:
                u_j = (c_ij * u_hat)
        u_j = u_j.view(batch_size, self.num_routes[0], self.num_routes[1], self.num_routes[2], -1)
        u_j = u_j.permute(0, 1, 4, 2, 3)
        return v_j.squeeze(1), self.out_conv(torch.norm(u_j, dim=2))
