import os
import pandas as pd
import numpy as np
from typing import Tuple
import torch

class StockLoader:
    def __init__(self, stock_path):
        self.stock_path = stock_path
        self.cache = {}

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

    def get_component(self, index_name:str, component_name:str):
        """
        Return dates and prices for a single component stock
        :param index_name: symbol for index
        :param component_name: symbol for component
        :return: np.array of (date, price) tuples
        """
        df = pd.read_csv(os.path.join(self.stock_path, index_name, f'{component_name.upper()}.csv'))
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

    def get_all_component_prices(self, index_name:str, split=True):
        if index_name in self.cache:
            return self.cache[index_name]

        component_names = self.get_component_names(index_name)
        prices = [self.get_component(index_name, c) for c in component_names]
        prices = self._crop(prices)
        self.cache[index_name] = prices

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

    def _crop(self, prices):
        """
        Trim prices to be the same length. Toss out histories which aren't long enough.
        This function takes a while and runtime can probably be improved.
        :param prices: prices to be trimmed
        :return: list of price histories
        """
        # find the most common length of stock price history
        unique, counts = np.unique(list(map(lambda x: len(x), prices)), return_counts=True)
        modal_len = unique[np.argmax(counts)]

        # throw out stocks with len < modal_len
        prices = list(filter(lambda x: len(x) >= modal_len, prices))

        # count dates
        dates = {}
        for series in prices:
            for day in series:
                dates[day[0]] = dates[day[0]] + 1 if day[0] in dates else 1

        for date in dates:
            if dates[date] < 0.9 * len(prices):
                dates[date] = None

        dates = {k:v for k,v in dates.items() if v is not None}

        cropped = []
        for series in prices:
            c = list(map(lambda x: x[1], filter(lambda y: y[0] in dates, series)))
            cropped.append(c)

        return cropped

if __name__ == '__main__':
    loader = StockLoader('stock_data')
    loader.get_all_component_prices(index_name='nya')