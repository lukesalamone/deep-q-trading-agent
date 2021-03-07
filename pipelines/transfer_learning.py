from typing import Tuple
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.models import Net
from utils.transfer_data import init_dataset
import torch

from utils.load_file import index_prices, component_prices, index_component_names

def compute_correlation(x: np.array, y: np.array) -> float:
    return np.corrcoef(x, y)[0,1]

def fake_prices(size, num_samples):
    return [torch.rand(size) for _ in range(num_samples)]

def train_mininet(size:int):
    model = Net(size=size)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    samples = fake_prices(size, 1000)

    for epoch in range(20):
        for x_i in samples:
            y_hat = model(x_i)

            loss = F.mse_loss(input=y_hat, target=x_i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate mse loss
        print(f'epoch {epoch} avg loss: {np.mean([F.mse_loss(X, model(X)).item() for X in samples])}')

    # run test
    test_samples = fake_prices(size, 100)
    losses = []

    for t in test_samples:
        y_hat = model(t)
        loss = F.mse_loss(y_hat, t)
        losses.append(loss.item())

    mse = np.mean(losses)
    print(f'average test loss: {mse}')

    return model



def mininet_pipeline():
    index = 'gspc'
    component_names = index_component_names(index)

    size = len(component_names)
    model = train_mininet(size)

    save_path = f'weights/mininet_{index}.pt'
    torch.save(model.state_dict(), save_path)

    component_mse = {}
    prices = np.array([component_prices(index, symbol) for symbol in component_names])
    prices_trans = prices.T
    prices_trans = torch.from_numpy(prices_trans)

    predictions_trans = model(prices_trans)
    predictions = predictions_trans.numpy().T

    for i, component in enumerate(component_names):
        actual = prices[i]
        predicted = predictions[i]
        mse = F.mse_loss(actual, predicted)
        component_mse[component] = mse

    print(component_mse)
    return model

if __name__ == '__main__':
    # mininet_pipeline()
    index = 'gspc'
    component_names = index_component_names(index)
    prices = np.array([component_prices(index, symbol) for symbol in component_names])
    lengths = list(map(lambda x: len(x), prices))
    print(lengths)