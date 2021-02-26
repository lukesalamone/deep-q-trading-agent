from typing import Tuple
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features=size, out_features=5, bias=True)
        self.out = nn.Linear(in_features=5, out_features=size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.out(x))
        return x

def train(model: Net, dataloaders: Tuple[DataLoader, DataLoader, DataLoader],
          epochs: int =10, learning_rate: float = 0.0001):

    train_loader, valid_loader, test_loader = dataloaders
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (X, Y) in enumerate(train_loader):
            # predict
            Y_hat = model(X)
            # compute MSE
            loss = F.mse_loss(input=Y_hat, target=Y)
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

    #TODO: ADD Performance logging, eg. MSE on Valid and Train

    return model


def compute_MSE(model: Net, X: np.array, Y: np.array):
    """
    :param model:
    :param X:
    :param Y:
    :return:
    """
    y_hat = model(X)
    return F.mse_loss(input=y_hat, target=Y)


def MSE_pipeline(model):
    pass

def compute_correlation(X: np.array, Y: np.array) -> float:
    return np.corrcoef(x=X, y=Y)

