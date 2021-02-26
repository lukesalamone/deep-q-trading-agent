import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from pipelines.build_batches import _load_from_file

class StockPrices(Dataset):
    def __init__(self, x: np.array, y: np.array):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

def _prices_to_dataloader(inputs: np.ndarray, targets: np.ndarray, batch_size: int) -> DataLoader:
    """
    :param inputs: prices of input stock
    :param targets: prices of target stock, usually the Index stock
    :param batch_size: batch size
    :return: DataLoader for SGD
    """
    # cut off any data that will create incomplete batches
    assert len(inputs) == len(targets)
    num_batches = len(inputs) // batch_size
    inputs = inputs[:num_batches * batch_size]
    targets = targets[:num_batches * batch_size]

    # create Dataset object and from it create data loader
    dataset = StockPrices(x=inputs, y=targets)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def _generate_fake_prices(size: int) -> Tuple[np.array, np.array, np.array]:
    """
    We use this to generate some fake data
    :param size:
    :return:
    """
    x = np.random.rand(size)

    train_idx = int(size*0.8)
    valid_idx = int(size*0.1)

    train = x[:train_idx]
    valid = x[train_idx:valid_idx]
    test = x[valid_idx:]

    return train, valid, test


def init_dataset(stock_path:str, index_path:str, batch_size:int, fake: bool = True
                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    :param stock_path: path to stock data
    :param index_path: path to index data
    :param batch_size: batch size
    :return: Tup of data loaders: (train, valid, test)
    """
    if fake:
        # WE GENERATE FAKE DATA
        x_train, x_valid, x_test = _generate_fake_prices(200)
        y_train, y_valid, y_test = _generate_fake_prices(200)
    else:
        # TODO: read component stock data properly. We don't have component stock data yet
        #  THIS ASSUMES THAT WE HAVE CREATED train, valid, test for each stock.
        #  USING SAME DATES AS WITH INDEX
        x_train, x_valid, x_test = _load_from_file(dataset='hsi')
        # load index stock data
        y_train, y_valid, y_test = _load_from_file('hsi')

    train_loader = _prices_to_dataloader(inputs=x_train, targets=y_train, batch_size=batch_size)
    valid_loader = _prices_to_dataloader(inputs=x_valid, targets=y_valid, batch_size=batch_size)
    test_loader = _prices_to_dataloader(inputs=x_test, targets=y_test, batch_size=batch_size)

    return (train_loader, valid_loader, test_loader)
