import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

NUMQ = 0
NUMDREG_AD = 1
NUMDREG_ID = 2


class DQN():
    def __init__(self, method):
        self.BUY = 0
        self.HOLD = 1
        self.SELL = 2

        # Set method
        self.method = method

        self.policy_net = None
        self.target_net = None

        # Set architecture
        if self.method == NUMQ:
            self.policy_net = NumQModel()
            self.target_net = NumQModel()
        elif self.method == NUMDREG_AD:
            self.policy_net = NumDRegModel(NUMDREG_AD)
            self.target_net = NumDRegModel(NUMDREG_AD)
        elif self.method == NUMDREG_ID:
            self.policy_net = NumDRegModel(NUMDREG_ID)
            self.target_net = NumDRegModel(NUMDREG_ID)

        # Make sure they start with the same weights
        self.transfer_weights()

    def transfer_weights(self):
        # TODO: Do we use TAU?
        # Update the target network, copying all weights and biases in DQN
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def action_index_to_value(self, action_index: int) -> int:
        """
        Given the index of the action, return the value
        :param action_index: action index (BUY=0, HOLD=1, SELL=2)
        :return: 1 - action index (BUY=1, HOLD=0, SELL=-1)
        """
        return 1 - action_index


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
    def __init__(self, method):
        super().__init__()

        # Set method
        self.method = method

        # Training step
        self.step = 0

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

        if self.step == 1:
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

    def set_step(self, s):
        self.step = s


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=1, out_features=5, bias=True)
        self.out = nn.Linear(in_features=5, out_features=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.out(x))
        return x