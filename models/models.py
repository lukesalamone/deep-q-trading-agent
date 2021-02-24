import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

class DQN():
    def __init__(self, architecture: nn.Module):
        self.BUY = 0
        self.HOLD = 1
        self.SELL = 2
        self.policy_net = architecture
        self.target_net = architecture

    def transfer_weights(self):
        # TODO: Do we use TAU?
        # Update the target network, copying all weights and biases in DQN
        self.target_net.load_state_dict(self.policy_net.state_dict())

class NumQModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=200, out_features=200, bias=True)
        self.fc2 = nn.Linear(in_features=200, out_features=100, bias=True)
        self.fc3 = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc_q = nn.Linear(in_features=50, out_features=3, bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        q = self.fc_q(F.relu(x))
        r = F.softmax(self.fc_q(F.sigmoid(x)))

        return q, r

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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # root
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
