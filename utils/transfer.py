import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class SimilarityNet(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        # root
        self.fc1 = nn.Linear(in_features=size, out_features=5, bias=True)
        self.out = nn.Linear(in_features=5, out_features=size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.out(x))
        return x