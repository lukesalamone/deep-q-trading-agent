import yfinance as yf
import sys

########################################
#          set variables here          #

INDEX_NAME = None
COMPONENTS_LIST = []

########################################

def main():
    for i, symbol in enumerate(COMPONENTS_LIST):
        symbol = f'{symbol}'
        print(f'{INDEX_NAME}: ({i+1}/{len(COMPONENTS_LIST)}) downloading symbol { symbol } . . .')
        data = yf.download(symbol, start="1987-01-01", end="2017-12-31")
        data.to_csv(f'../raw/{INDEX_NAME}/{symbol}.csv')

if __name__ == '__main__':

    # parameters can be specified by command line or in file
    if not INDEX_NAME or not COMPONENTS_LIST:
        print('please specify index name and components above')
    main()