import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

class NumQDRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        # root
        self.fc1 = nn.Linear(in_features=200, out_features=100, bias=True)
        # action branch
        self.fc2_act = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc3_act = nn.Linear(in_features=50, out_features=20, bias=True)
        self.fc_q = nn.Linear(in_features=20, out_features=3, bias=True)
        # number branch
        self.fc2_num = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc3_num = nn.Linear(in_features=50, out_features=20, bias=True)
        self.fc_r = nn.Linear(in_features=20, out_features=3, bias=True)

    def forward(self, x:Tensor):
        x = F.relu(self.fc1(x))

        # action branch
        x_act = F.relu(self.fc2_act(x))
        x_act = F.relu(self.fc3_act(x_act))
        q = self.fc_q(x_act)

        # number branch
        x_num = F.relu(self.fc2_num(x))
        x_num = F.sigmoid(self.fc3_num(x_num))
        r = F.softmax(self.fc_r(x_num))

        return q, r
