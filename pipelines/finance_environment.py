from collections import deque
from typing import Tuple
import torch
from torch import Tensor
import random
import numpy as np
import pandas as pd
import yaml
import os

from .build_batches import load_prices
from utils.rewards import compute_reward, compute_profit

class FinanceEnvironment:
    def __init__(self, price_history, mem_cap, lookback):
        self.price_history = price_history
        self.time_step = 0
        self.profits = []
        self.replay_memory = ReplayMemory(capacity=mem_cap)
        self.lookback = lookback

        price_col_name = self.price_history.columns[1]
        prices = torch.tensor(self.price_history[price_col_name].to_numpy())

        self.today_prices = torch.Tensor(prices[1:])
        self.yesterday_prices = torch.Tensor(prices[:-1])
        self.states = self.today_prices - self.yesterday_prices

    def start_episode(self):
        self.time_step = 0
        self.episode_losses = []
        self.episode_rewards = []
        self.episode_profit = 0

        self.cur_state = self.states[0:self.lookback]
        self.next_state = self.states[1:self.lookback + 1]
        self.cur_price = self.today_prices[self.lookback - 1]
        self.cur_prev_price = self.yesterday_prices[self.lookback]
        self.init_price = self.today_prices[0]


    def step(self):
        """
        Move forward one step in time and advance all relevant variables
        :return: state at time_step and done flag
        """
        self.cur_state = self.states[self.time_step:self.lookback+self.time_step]
        self.next_state = self.states[self.time_step + 1:self.lookback + self.time_step + 1]
        self.cur_price = self.today_prices[self.lookback + self.time_step- 1]
        self.cur_prev_price = self.yesterday_prices[self.lookback + self.time_step]
        self.init_price = self.today_prices[self.time_step]

        # TODO Simon may have better logic here
        done = len(self.today_prices) < self.time_step

        self.time_step += 1

        return self.cur_state, done

    def add_profit(self, profit):
        self.episode_profit += profit

    def add_loss(self, loss):
        self.episode_losses.append(loss)

    def profit_and_reward(self, action, num):
        self.cur_action = action
        self.cur_num = num

        profit = compute_profit(num_t=num,
                       action_value=action,
                       price=self.cur_price,
                       prev_price=self.cur_prev_price)

        reward = compute_reward(num_t=num,
                                action_value=action,
                                price=self.cur_price,
                                prev_price=self.cur_prev_price,
                                init_price=self.init_price)

        self.cur_reward = reward

        self.episode_rewards.append(reward)
        self.episode_profit += profit
        return profit, reward


    def update_replay_memory(self):
        self.replay_memory.update(
            (self.cur_state, self.cur_action, self.cur_reward, self.cur_next_state)
        )

    def on_episode_end(self):
        avg_loss = sum(self.episode_losses) / len(self.episode_losses)
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        return avg_loss, avg_reward, self.episode_profit

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def update(self, transition: Tuple[Tensor, int, Tensor, float]):
        """
        Saves a transition
        :param transition: (state, action_index, next_state, reward)
        :return:
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def make_env(index, symbol, mem_cap, lookback):
    prices = load_prices(index, symbol)
    return FinanceEnvironment(prices, mem_cap, lookback)


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
