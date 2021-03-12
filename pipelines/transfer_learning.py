from torch import optim
from torch.nn import functional as F
from models.models import StonksNet
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from utils.load_file import *

def fake_prices(size, num_samples):
    return [torch.rand(size) for _ in range(num_samples)]


def correlation_pipeline():
    index = 'nyse'
    component_names = index_component_names(index)
    prices_i = load_index_prices('^NYA')

    component_corr = {}

    for component in component_names:
        prices_c = component_prices(index, component)
        component_corr[component] = np.corrcoef(prices_i, prices_c)[0, 1]

    print(component_corr)
    outpath = f'stonks/relationships/correlation/{index}.csv'
    df = pd.DataFrame(list(component_corr.items()), columns=['stonk', 'correlation'])
    df.to_csv(outpath)


def train_stonksnet(prices:Tensor):
    num_days, num_components = tuple(prices.size())
    model = StonksNet(size=num_components)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(20):
        # train day by day
        losses = []
        for day in prices:
            optimizer.zero_grad()
            output = model(day)
            loss = criterion(output, day)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'epoch {epoch} avg loss: {np.mean(losses)}')
    return model


def mininet_pipeline():
    index = 'nasdaq'
    load = False
    model_path = f'weights/mininet_{index}.pt'

    # load component prices
    # each row is a trading day, each column is a component
    component_prices = load_component_prices(index)

    if load:
        model = torch.load(model_path)
    else:
        model = train_stonksnet(component_prices)
        torch.save(model, model_path)

    predicted_prices = model(component_prices)
    component_names = index_component_names(index)
    component_mse = {}

    # transpose actual and predicted so that each row is now a component
    component_prices = torch.transpose(component_prices, dim0=0, dim1=1)
    predicted_prices = torch.transpose(predicted_prices, dim0=0, dim1=1)

    for i, symbol in enumerate(component_names):
        predicted = predicted_prices[i]
        actual = component_prices[i]
        loss = F.mse_loss(input=predicted, target=actual)
        component_mse[symbol] = loss.item()

    outpath = f'stonks/relationships/stonksnet/{index}.csv'
    df = pd.DataFrame(list(component_mse.items()), columns=['stonk', 'MSE'])
    df.to_csv(outpath)

    return model

def gather_groups():
    group_sizes = {'djia':4, 'gspc':6, 'nasdaq':6, 'nyse':8}
    groups = {}


    for index in group_sizes:
        size = group_sizes[index]
        hs = int(size/2)
        group = {'correlation': {}, 'mse': {}}
        correlations = load_relationship_info('correlation', index)
        correlations = correlations.sort_values(by=['correlation'], ascending=False).to_numpy()
        correlations = list(map(lambda x: x[1], correlations))

        # high correlation, low correlation, half high + half low
        group['correlation']['high'] = correlations[0:size]
        group['correlation']['low'] = correlations[-size:]
        group['correlation']['highlow'] = correlations[-hs:] + correlations[0:hs]

        mse = load_relationship_info('stonksnet', index)
        mse = mse.sort_values(by=['MSE'], ascending=False).to_numpy()
        mse = list(map(lambda x: x[1], mse))

        group['mse']['high'] = mse[0:size]
        group['mse']['low'] = mse[-size:]
        group['mse']['highlow'] = mse[-hs:] + mse[0:hs]

        groups[index] = group

    return groups


if __name__ == '__main__':
    groups = gather_groups()
    print(groups)
