import numpy as np
import pandas as pd
import yaml

from .build_batches import load_prices



class FinanceEnvironment:
    def __init__(self, prices):
        self.prices = prices
        self.time_step = 0

    def step(self):
        self.time_step += 1

def make_env(symbol):
    prices = load_prices(symbol)
    return FinanceEnvironment(prices)


# Get all config values and hyperparameters
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile)

def _read_csv(index: str, ticker: str) -> pd.DataFrame:
    """
    input an index and the ticker. read csv data
    :param index:
    :param ticker:
    :return:
    """
    file_name = f"{index}/{ticker}.csv"
    file_path = os.path.join(config["DATA_DIRECTORY"], file_name)
    return pd.read_csv(file_path)

def _get_state(prices: pd.DataFrame, step: int, n: int, price_col:str, end_date: str):
    """
    :param prices: price data for this stock
    :param step: Timestep t
    :param n: LOOKBACK
    :param price_col: column corresponding to closing trade price in the df (differs for eurostoxx) #TODO: Homogenize data
    :param end_date: date at which we end the episode
    :return:
    """
    # initialize end
    end = False

    #TODO: PAD THIS DATAFRAME. IF LOOKBACK > STEP, WE ARE FUCKED

    # look up price in df at index step
    p_t = prices[price_col].at[step]
    # look up prev price in df at index step-1
    p_t_prev = prices[price_col].at[step-1]
    # look up init price in df at index step-n
    p_init = prices[price_col].at[step-n]
    # we take the difference in prices along the column for indexes step-n to step
    s_t = prices[price_col].diff(1).loc[step-n:step]
    # check the date at index step.
    # If it's the end date, we are at the end of the episode
    if prices['Date'].at[step] == end_date:
        end = True
    return p_t, p_t_prev, p_init, np.array(s_t), end
