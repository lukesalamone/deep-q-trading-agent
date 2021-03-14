import yfinance as yf
import sys
import json
import os

########################################
#          set variables here          #

METADATA_PATH = "../stock_data/metadata.json"
START_DATE = "1991-01-01"
END_DATE = "2021-01-01"


########################################

def download_index(name, index, components_list):
    components_list.append(index)
    dir_name = index[1:].lower()
    os.mkdir(f'../raw/{dir_name}')

    for i, symbol in enumerate(components_list):
        symbol = f'{symbol}'
        print(f'{name}: ({i+1}/{len(components_list)}) downloading symbol { symbol } . . .')
        data = yf.download(symbol, start="1987-01-01", end="2017-12-31")
        data.to_csv(f'../raw/{dir_name}/{symbol}.csv')


def main():
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    for index in metadata:
        index_name, index_symbol, components = index.values()
        print(f'\n\n-------- DOWNLOADING COMPONENTS FOR {index_name} --------\n\n')
        download_index(index_name, index_symbol, components)


if __name__ == '__main__':
    main()