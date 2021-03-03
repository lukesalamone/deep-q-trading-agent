import numpy as np
import pandas as pd
import yaml
import os

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

def _get_state(prices: pd.DataFrame, step: int, n: int, price_col:str, end_date: str, start_date: str=''):
    """
    :param prices: price data for this stock
    :param step: Timestep t
    :param n: LOOKBACK
    :param price_col: column corresponding to closing trade price in the df (differs for eurostoxx) #TODO: Homogenize data
    :param end_date: date at which we end the episode
    :param start_date: date at which we end the episode
    :return:
    """
    # TODO: DISCUSS FACT THAT THIS REQUIRES KEEPING FULL CSV IN MEMORY
    #  Maybe we can just return restricted dataframe from read_csv
    #  eg. df = read_csv => smaller_df = keep_dates(df)
    #  also if we do this we dont need the dates to determine the end of an episode

    # initialize end
    end = False
    # we get the start index based on start_date.
    # this is useful if you are testing, then set start date to that date.
    start_index = prices['Date'].get_loc[start_date] if start_date else 0

    # if lookback > n we are in trouble,
    # so we pad this dataframe with rows identical to the first row
    padding = prices.at[0]*n
    prices = pd.concat([padding, prices], axis=1, ignore_index=True)

    # update step value because of padding and start index
    step = step + n + start_index

    # look up price, prev price, init price in df at indices step, step-1, step-n
    p_t = prices[price_col].at[step]
    p_t_prev = prices[price_col].at[step-1]
    p_init = prices[price_col].at[step-n]

    # take the difference in prices along the column for indexes step-n to step
    s_t = prices[price_col].diff(1).loc[step-n:step]

    # check the date at index step.
    # If it's the end date, we are at the end of the episode
    if prices['Date'].at[step] == end_date:
        end = True
    return p_t, p_t_prev, p_init, np.array(s_t), end
