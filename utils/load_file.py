import os
import pandas as pd
import numpy as np
from typing import Tuple
import torch

class StockLoader:
    def __init__(self, stock_path):
        self.stock_path = stock_path

    def index_prices(self, index_name:str):
        """
        Load prices for index only (not components)
        :param index_name: symbol for index
        :return: list of prices
        """
        return self.get_component_prices(index_name, f'^{index_name.upper()}.csv')

    def get_component_prices(self, index_name:str, component_name:str):
        """
        Return prices for a component of an index.
        :param index_name: symbol for index
        :param component_name: symbol for component
        :return: list of prices
        """
        df = pd.read_csv(os.path.join(self.stock_path, index_name, f'{component_name.upper()}.csv'))
        df = df[df.columns[1]].astype('float64')
        return df.to_numpy()

    def get_component_names(self, index_name: str):
        """
        Return list of symbols of components
        :param index_name:
        :return:
        """
        dir = os.path.join(self.stock_path, index_name)
        files = os.listdir(dir)
        files = map(lambda x: x[:-4], files)
        files = filter(lambda x: not x.startswith('^'), files)
        return list(files)

    def get_index_prices(self, index_name: str):
        df = pd.read_csv(os.path.join(self.stock_path, 'index_data', f'^{index_name.upper()}.csv'))
        df = df[df.columns[1]].astype('float64')
        return df.to_numpy()

    def get_all_component_prices(self, index_name: str, split=True):
        component_names = self.get_component_names(index_name)
        prices = [self.get_component_prices(index_name, c) for c in component_names]

        if split:
            prices = [self.train_test_splits(p)[0] for p in prices]

        return torch.transpose(torch.tensor(prices), dim0=0, dim1=1)

    def train_test_splits(self, prices:np.array) -> Tuple[np.array, np.array]:
        """
        Split prices into train, test: 2/3, 1/3
        :param prices:
        :return:
        """
        train_idx = int(len(prices) * (2 / 3))

        train = prices[:train_idx]
        test = prices[train_idx:]

        return train, test

    def load_relationship_info(self, type, index):
        path = os.path.join(self.stock_path, 'relationships', type, f'{index}.csv')
        return pd.read_csv(path)



if __name__ == '__main__':
    pass
