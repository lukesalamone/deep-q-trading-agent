from typing import Tuple
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.models import Net

def compute_correlation(X: np.array, Y: np.array) -> float:
    return np.corrcoef(x=X, y=Y)

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
            # for param in model.parameters():
            #     param.grad.data.clamp_(-1, 1)
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

def MSE_pipeline(index, component_stocks):
    # TODO: READ Data for an index and component stocks
    stock_MSEs = { }
    for stock in component_stocks:
        dataloaders = init_dataset(stock_path='PLACEHOLDER', index_path='PLACEHOLDER', batch_size=32, fake=True)
        _, valid, _ = dataloaders
        # Train model
        model = Net()
        # place holders
        X = np.array([0])
        Y = np.array([0])
        model = train(model=model, dataloaders=dataloaders)
        stock_MSEs['STOCK stock'] = compute_MSE(model=model, X=X, Y=Y)
    return stock_MSEs
