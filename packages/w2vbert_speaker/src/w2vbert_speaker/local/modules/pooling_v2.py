"""Pooling layers used by speaker embedding models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GSP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.expansion = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)


class ASP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.expansion = 2
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x.transpose(1, 2)).transpose(1, 2)
        mu = torch.sum(x * weights, dim=1)
        sigma = torch.sqrt((torch.sum((x**2) * weights, dim=1) - mu**2).clamp(min=1e-5))
        return torch.cat([mu, sigma], dim=1)
