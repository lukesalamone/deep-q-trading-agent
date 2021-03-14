from torch import optim
from torch.nn import functional as F
from models.models import StonksNet
import torch
import torch.nn as nn
from torch import Tensor
import pandas as pd
import numpy as np
import json

from utils.load_file import StockLoader

# Get all config values and hyperparameters
# with open("config.yml", "r") as ymlfile:
#     config = yaml.load(ymlfile, Loader=yaml.FullLoader)

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


def measure_correlation(index:str, outpath:str, loader:StockLoader):
    component_names = loader.get_component_names(index)
    prices_i = loader.get_index_prices(index)

    component_corr = {}

    for component in component_names:
        prices_c = loader.get_component_prices(index, component)
        component_corr[component] = np.corrcoef(prices_i, prices_c)[0, 1]

    print(component_corr)
    outpath = f'{outpath}/correlation/{index}.csv'
    df = pd.DataFrame(list(component_corr.items()), columns=['symbol', 'correlation'])
    df.to_csv(outpath)


def measure_autoencoder_mse(index:str, outpath:str, loader:StockLoader):
    load = False
    model_path = f'weights/mininet_{index}.pt'
    index = index.lower().replace('^', '')

    # load component prices
    # each row is a trading day, each column is a component
    component_prices = loader.get_component_prices(index)

    if load:
        model = torch.load(model_path)
    else:
        model = train_stonksnet(component_prices)
        torch.save(model, model_path)

    predicted_prices = model(component_prices)
    component_names = loader.get_component_names(index)
    component_mse = {}

    # transpose actual and predicted so that each row is now a component
    component_prices = torch.transpose(component_prices, dim0=0, dim1=1)
    predicted_prices = torch.transpose(predicted_prices, dim0=0, dim1=1)

    for i, symbol in enumerate(component_names):
        predicted = predicted_prices[i]
        actual = component_prices[i]
        loss = F.mse_loss(input=predicted, target=actual)
        component_mse[symbol] = loss.item()

    outpath = f'{outpath}/mse/{index}.csv'
    df = pd.DataFrame(list(component_mse.items()), columns=['stonk', 'MSE'])
    df.to_csv(outpath)

    return model

def gather_groups(group_sizes:dict, loader:StockLoader):
    groups = {}

    for index in group_sizes:
        size = group_sizes[index]
        hs = int(size/2)
        correlations = loader.load_relationship_info('correlation', index)
        correlations = correlations.sort_values(by=['correlation'], ascending=False).to_numpy()
        correlations = list(map(lambda x: x[1], correlations))

        mse = loader.load_relationship_info('mse', index)
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
    loader = StockLoader(stock_path=STOCK_PATH)

    for index in metadata:
        index_name, index_symbol, components, num = index.values()
        measure_autoencoder_mse(index=index_symbol, outpath=OUT_PATH, loader=loader)
        measure_correlation(index=index_symbol, outpath=OUT_PATH, loader=loader)

    groups = gather_groups({x['symbol'][1:].lower():x['tl_size'] for x in metadata}, loader)

