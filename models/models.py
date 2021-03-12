import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

NUMQ = 0
NUMDREG_AD = 1
NUMDREG_ID = 2

ACT_MODE = 0
NUM_MODE = 1
FULL_MODE = 2

torch.set_default_dtype(torch.float64)

class DQN():
    def __init__(self, method):
        self.BUY = 0
        self.HOLD = 1
        self.SELL = 2
        
        # Set method and mode
        self.method = method
        self.mode = FULL_MODE

        self.policy_net = None
        self.target_net = None

        # Set architecture
        if self.method == NUMQ:
            self.policy_net = NumQModel()
            self.target_net = NumQModel()
        elif self.method == NUMDREG_AD:
            self.policy_net = NumDRegModel(NUMDREG_AD, self.mode)
            self.target_net = NumDRegModel(NUMDREG_AD, self.mode)
        elif self.method == NUMDREG_ID:
            self.policy_net = NumDRegModel(NUMDREG_ID, self.mode)
            self.target_net = NumDRegModel(NUMDREG_ID, self.mode)

        # Make sure they start with the same weights
        # self.hard_update()

    def hard_update(self):
        # Update the target network, copying all weights and biases in DQN
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self, tau: float):
        """
        Soft update model parameters.
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        :param tau: interpolation parameter
        :return:
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
    
    def set_mode(self, m):
        self.mode = m
        self.policy_net.mode = self.mode
        self.target_net.mode = self.mode


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
        r = F.softmax(self.fc_q(torch.sigmoid(x)))

        return q, r


class NumDRegModel(nn.Module):
    def __init__(self, method, mode):
        super().__init__()

        # Set method
        self.method = method

        # Mode for training
        self.mode = mode

        # root
        self.fc1 = nn.Linear(in_features=200, out_features=100, bias=True)

        # action branch
        self.fc2_act = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc3_act = nn.Linear(in_features=50, out_features=20, bias=True)
        self.fc_q = nn.Linear(in_features=20, out_features=3, bias=True)

        # number branch
        self.fc2_num = nn.Linear(in_features=100, out_features=50, bias=True)
        self.fc3_num = nn.Linear(in_features=50, out_features=20, bias=True)
        self.fc_r = nn.Linear(in_features=20, out_features=(3 if self.method == NUMDREG_AD else 1), bias=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Root
        x = F.relu(self.fc1(x))

        # Action branch
        x_act = F.relu(self.fc2_act(x))
        x_act = F.relu(self.fc3_act(x_act))
        q = self.fc_q(x_act)

        if self.mode == ACT_MODE:
            # Number branch based on q values
            r = F.softmax(self.fc_q(torch.sigmoid(x_act)))
        else:
            # Number branch
            x_num = F.relu(self.fc2_num(x))
            x_num = torch.sigmoid(self.fc3_num(x_num))
            # Output layer depends on method
            if self.method == NUMDREG_ID:
                r = torch.sigmoid(self.fc_r(x_num))
            else:
                r = F.softmax(self.fc_r(x_num))

        return q, r

class StonksNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.fc1 = nn.Linear(in_features=size, out_features=5, bias=True)
        self.out = nn.Linear(in_features=5, out_features=size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.out(x))
        return x