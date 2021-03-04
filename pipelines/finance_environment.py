from collections import deque
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
import random
import pandas as pd
import yaml

from .build_batches import load_prices

# Get all config values and hyperparameters
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile)

class FinanceEnvironment:
    def __init__(self, price_history: pd.DataFrame, index: str, dataset:str):

        self.start_date, self.end_date = config["INDEX_SPLITS"][index][dataset]
        self.lookback = config["LOOKBACK"]

        self.price_history = price_history
        self.date_column, self.price_column = self.price_history.columns
        # self.price_history[self.price_column] = self.price_history[self.price_column].astype(dtype=np.float32)
        self.init_prices()

        # BUY, HOLD, SELL, = (1, 0, -1
        self.action_space = (1, 0, -1)

        # pad using start date and set time_step at lookback + start_index
        self.profits = []
        self.replay_memory = ReplayMemory(capacity=config["MEMORY_CAPACITY"])

    def init_prices(self):
        # we get the start index based on start_date.
        self.start_index = self.price_history[self.price_history[self.date_column] == self.start_date].index.values[0]

        # if lookback > t we are in trouble,
        # so we pad this dataframe with rows identical to the first row and reset the index
        pad = pd.concat([self.price_history.iloc[[0]]] * self.lookback)
        self.price_history = pd.concat([pad, self.price_history], ignore_index=True)
        # we create a pd.Series where at t, we have price - price_prev
        # we backfill to avoid having a NaN in the first value
        # all padded timesteps will have 0
        price_differences = self.price_history[self.price_column].diff(1).fillna(method='backfill')
        self.price_differences = torch.Tensor(price_differences)

    def start_episode(self):
        self.episode_losses = []
        self.episode_rewards = []
        self.episode_profit = 0
        # update step value because of padding and start index
        self.time_step = self.lookback + self.start_index

    def step(self):
        # look up price, prev price, init price in df at indices timestep, timestep-1, timestep-lookback
        self.price = self.price_history[self.price_column].at[self.time_step]
        self.prev_price = self.price_history[self.price_column].at[self.time_step - 1]
        self.init_price = self.price_history[self.price_column].at[self.time_step - self.lookback]

        # get the state as the day to day price differences from timestep-n to timestep
        self.state = self.price_differences[self.time_step - self.lookback:self.time_step]
        assert self.state.size() == torch.Size([self.lookback])

        # check the date at index step.
        # If it's the end date, we are at the end of the episode
        # end = self.price_history[self.date_column].at[self.time_step] == self.end_date
        # TODO: CHANGE DATES TO WORK WITH DATETIME
        #  Use >= in case dates are missing
        end = self.price_history[self.date_column].at[self.time_step] == self.end_date

        if end:
            self.next_state = torch.Tensor([])
        else:
            # get the next state
            self.next_state = self.price_differences[self.time_step - self.lookback + 1:self.time_step + 1]
            assert self.next_state.size() == torch.Size([self.lookback])

        # increment timestep
        self.time_step += 1

        return self.state, end


    def compute_profit_and_reward(self, action_index: int, num: float):
        self.action = action_index
        self.num = num

        action_value = self.action_space[action_index]

        profit = _profit(num=num,
                         action_value=action_value,
                         price=self.price,
                         prev_price=self.prev_price)

        reward = _reward(num=num,
                         action_value=action_value,
                         price=self.price,
                         prev_price=self.prev_price,
                         init_price=self.init_price)

        self.reward = reward

        self.episode_rewards.append(reward)
        self.episode_profit += profit

        return profit, reward

    def add_loss(self, loss):
        self.episode_losses.append(loss)

    def update_replay_memory(self):
        # checks if torch tensor is empty
        if self.next_state.numel():
            self.replay_memory.update(
                (self.state, self.action, self.reward, self.next_state)
            )

    def on_episode_end(self):
        avg_loss = sum(self.episode_losses) / len(self.episode_losses)
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        return avg_loss, avg_reward, self.episode_profit


def _reward(num: float, action_value: int, price: float, prev_price: float, init_price: float) -> float:
    """
    :param num:
    :param action_value:
    :param price:
    :param prev_price:
    :param init_price:
    :return:
    """
    reward = 1 + action_value * (price - prev_price) / prev_price
    reward = num * reward * prev_price / init_price
    return reward


def _profit(num: float, action_value: int, price: float, prev_price: float) -> float:
    """
    :param num:
    :param action_value:
    :param price:
    :param prev_price:
    :return:
    """
    return num * action_value * (price - prev_price) / prev_price


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def update(self, transition: Tuple[Tensor, int, float, Tensor]):
        """
        Saves a transition
        :param transition: (state, action_index, reward, next_state)
        :return:
        """
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def make_env(index: str, symbol: str, dataset:str):
    """
    :param index: stock index, eg. gspc, hsi, etc.
    :param symbol: Ticker, eg. '^GSPC', 'APX'
    :param dataset: 'train', 'valid', 'test'
    :return:
    """
    prices = load_prices(index=index, symbol=symbol)
    return FinanceEnvironment(price_history=prices, index=index, dataset=dataset)
