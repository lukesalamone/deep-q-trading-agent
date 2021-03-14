import pandas as pd
from pandas import DataFrame
import yfinance as yf
import sys
import json
import os
from pathlib import Path

METADATA_PATH = "stock_data/metadata.json"
START_DATE = "1991-01-01"
END_DATE = "2021-01-01"
OUT_PATH = 'stock_data'

def download_index(name, index, components_list):
    components_list.append(index)
    dir_name = index[1:].lower()
    dir_path = f'{OUT_PATH}/{dir_name}'

    if not Path(dir_path).exists():
        os.mkdir(dir_path)

    for i, symbol in enumerate(components_list):
        symbol = f'{symbol}'
        print(f'{name}: ({i+1}/{len(components_list)}) downloading symbol { symbol } . . .')

        # download csv from yfinance
        data = yf.download(symbol, start=START_DATE, end=END_DATE)

        # fix missing data points
        data = data['Adj Close'].fillna(method='ffill')

        data.to_csv(f'{OUT_PATH}/{dir_name}/{symbol}.csv')


def download_files():
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    for index in metadata:
        index_name, index_symbol, components, _ = index.values()
        print(f'\n\n-------- DOWNLOADING COMPONENTS FOR {index_name} --------\n\n')
        download_index(index_name, index_symbol, components)


if __name__ == '__main__':
    # downloads stock data files and adds them to ../raw/
    download_files()


