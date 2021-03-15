import yaml
from torch import optim
from torch.nn import functional as F
from models.models import StonksNet
import torch
import torch.nn as nn
from torch import Tensor
import pandas as pd
import numpy as np
import json
import os

from utils.load_file import StockLoader

METADATA_PATH = "stock_data/metadata.json"
OUT_PATH = 'stock_data/relationships'


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


def measure_correlation(outpath:str, loader:StockLoader):
    prices, component_names = loader.get_all_component_prices(include_index=True, transpose=False)
    prices = prices.numpy()

    component_names = component_names[:-1]
    prices_c = prices[:-1]
    prices_i = prices[-1]

    component_corr = {}

    # iterate over each component and measure correlation coef
    for name, prices in zip(component_names, prices_c):
        component_corr[name] = np.corrcoef(prices_i, prices)[0, 1]

    print(component_corr)
    outpath = f'{outpath}/correlation'
    df = pd.DataFrame(list(component_corr.items()), columns=['symbol', 'correlation'])

    os.makedirs(outpath, exist_ok=True)
    df.to_csv(f'{outpath}/{loader.index}.csv')


def measure_autoencoder_mse(outpath:str, loader:StockLoader):
    load = False
    index = loader.index
    model_path = f'weights/mininet_{index}.pt'

    # load component prices and included symbols
    # each row is a trading day, each column is a component
    component_prices, symbols = loader.get_all_component_prices(index)

    if load:
        model = torch.load(model_path)
    else:
        print(f'training autoencoder on {index}')
        model = train_stonksnet(component_prices)
        torch.save(model, model_path)

    predicted_prices = model(component_prices)
    component_mse = {}

    # transpose actual and predicted so that each row is now a component
    component_prices = torch.transpose(component_prices, dim0=0, dim1=1)
    predicted_prices = torch.transpose(predicted_prices, dim0=0, dim1=1)

    for i, symbol in enumerate(symbols):
        predicted = predicted_prices[i]
        actual = component_prices[i]
        loss = F.mse_loss(input=predicted, target=actual)
        component_mse[symbol] = loss.item()

    outpath = f'{outpath}/mse'
    df = pd.DataFrame(list(component_mse.items()), columns=['stonk', 'MSE'])

    os.makedirs(outpath, exist_ok=True)
    df.to_csv(f'{outpath}/{index}.csv')

    return model


def gather_groups():
    with open("config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    stock_path = config['STOCK_DATA_PATH']

    with open(os.path.join(stock_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    group_sizes = {x['symbol'][1:].lower(): x['tl_size'] for x in metadata}

    groups = {}

    for index in group_sizes:
        loader = StockLoader(index_name=index)
        size = group_sizes[index]
        hs = int(size/2)
        correlations = loader.load_relationship_info('correlation')
        correlations = correlations.sort_values(by=['correlation'], ascending=False).to_numpy()
        correlations = list(map(lambda x: x[1], correlations))

        mse = loader.load_relationship_info('mse')
        mse = mse.sort_values(by=['MSE'], ascending=False).to_numpy()
        mse = list(map(lambda x: x[1], mse))

        # high correlation, low correlation, half high + half low
        groups[index] = {
            'correlation': {
                'high': correlations[0:size],
                'low': correlations[-size:],
                'highlow': correlations[-hs:] + correlations[0:hs]
            }, 'mse': {
                'high': mse[0:size],
                'low': mse[-size:],
                'highlow': mse[-hs:] + mse[0:hs]
            }
        }

    return groups


if __name__ == '__main__':
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    STOCK_PATH = 'stock_data'

    # for index in metadata:
    #     loader = StockLoader(index_name=index['symbol'], stock_path=STOCK_PATH)
    #     index_name, index_symbol, components, num = index.values()
    #     measure_autoencoder_mse(outpath=OUT_PATH, loader=loader)
    #     measure_correlation(outpath=OUT_PATH, loader=loader)

    groups = gather_groups()

