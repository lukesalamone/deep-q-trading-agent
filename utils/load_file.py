import os
import pandas as pd
import numpy as np
from typing import Tuple
import torch


class StockLoader:
    def __init__(self, index_name:str, stock_path:str='stock_data'):
        """
        A StockLoader is a helper class to assist in loading stock data or relationship info.
        This class is intended to be used only with one index.
        :param stock_path: the path where stock data is stored
        :param index_name: the index to work with
        """
        self.stock_path = stock_path
        self.index = index_name.replace('^', '').lower()

    def index_prices(self):
        """
        Load prices for index only (not components)
        :return: list of prices
        """
        return self.get_component_prices(f'^{self.index.upper()}')

    def get_component_prices(self, component_name:str):
        """
        Return prices for a component of an index.
        :param component_name: symbol for component
        :return: list of prices
        """
        df = pd.read_csv(os.path.join(self.stock_path, self.index, f'{component_name.upper()}.csv'))
        df = df[df.columns[1]].astype('float64')
        return df.to_numpy()

    def get_component(self, component_name:str):
        """
        Return dates and prices for a single component stock
        :param component_name: symbol for component
        :return: np.array of (date, price) tuples
        """
        df = pd.read_csv(os.path.join(self.stock_path, self.index, f'{component_name.upper()}.csv'))
        return df.to_numpy()

    def get_component_names(self):
        """
        Return list of symbols of components
        :return:
        """
        directory = os.path.join(self.stock_path, self.index)
        files = os.listdir(directory)
        files = map(lambda x: x[:-4], files)
        files = filter(lambda x: not x.startswith('^'), files)
        return list(files)

    def get_index_prices(self):
        df = pd.read_csv(os.path.join(self.stock_path, self.index, f'^{self.index.upper()}.csv'))
        df = df[df.columns[1]].astype('float64')
        return df.to_numpy()

    def get_all_component_prices(self, split:bool=True, include_index:bool=False, transpose:bool=True):
        component_names = self.get_component_names()

        if include_index:
            component_names.append(f'^{self.index.upper()}')

        prices = [self.get_component(c) for c in component_names]
        prices, symbols = self._crop(prices, component_names)

        if split:
            prices = [self.train_test_splits(p)[0] for p in prices]

        prices = torch.tensor(prices)

        if not transpose:
            return prices, symbols

        return torch.transpose(prices, dim0=0, dim1=1), symbols

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

    def load_relationship_info(self, relationship_type):
        path = os.path.join(self.stock_path, 'relationships', relationship_type, f'{self.index}.csv')
        return pd.read_csv(path)

    def _crop(self, prices, component_names):
        """
        Trim prices to be the same length. Toss out histories which aren't long enough.
        This function takes a while and run time can probably be improved.
        :param prices: prices to be trimmed
        :param component_names list of symbols corresponding to prices
        :return: list of price histories, symbols not discarded
        """
        # find the most common length of stock price history
        unique, counts = np.unique(list(map(lambda x: len(x), prices)), return_counts=True)
        modal_len = unique[np.argmax(counts)]

        # throw out stocks with len < modal_len
        prices = list(filter(lambda x: len(x[0]) >= modal_len, zip(prices, component_names)))
        prices, symbols = zip(*prices)

        print(f'discarding price histories under length '
              f'{modal_len} ({len(component_names) - len(symbols)} symbols discarded)')

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

        return cropped, symbols


if __name__ == '__main__':
    loader = StockLoader(index_name='dji', stock_path='stock_data')
    loader.get_all_component_prices()