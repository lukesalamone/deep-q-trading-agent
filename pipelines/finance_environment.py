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

# Get all config values and hyperparameters
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile)

class FinanceEnvironment:
    def __init__(self, price_history: pd.DataFrame, mem_cap:int, index: str, task:str, lookback: int):
        self.start_date, self.end_date = config["INDEX_SPLITS"][index][task]
        self.price_history = price_history
        self.lookback = lookback
        self.action_space = (-1, 0, 1)

        # pad using start date and set time_step at lookback + start_index
        self.profits = []
        self.replay_memory = ReplayMemory(capacity=mem_cap)

        self.date_column, self.price_column = self.price_history.columns
        self.init_prices()
        self.init_episode()

    def init_prices(self):
        # we get the start index based on start_date.
        # this is useful if you are testing, then set start date to that date.
        start_index = self.price_history['Date'].get_loc[self.start_date]

        # if lookback > t we are in trouble,
        # so we pad this dataframe with rows identical to the first row
        padding = self.price_history.at[0] * self.lookback
        self.price_history = pd.concat([padding, self.price_history], axis=1, ignore_index=True)
        self.historical_price_differences = self.price_history[self.price_column].diff(1)

        # update step value because of padding and start index
        self.time_step = self.lookback + start_index

    def init_episode(self):
        self.episode_losses = []
        self.episode_rewards = []
        self.episode_profit = 0

    def step(self):
        # look up price, prev price, init price in df at indices timestep, timestep-1, timestep-lookback
        self.price = self.price_history[self.price_column].at[self.time_step]
        self.prev_price = self.price_history[self.price_column].at[self.time_step - 1]
        self.init_price = self.price_history[self.price_column].at[self.time_step - self.lookback]

        # get the state as the day to day price differences from timestep-n to timestep
        state = self.historical_price_differences.loc[self.time_step - self.lookback : self.time_step]
        # get the next state
        next_state = self.historical_price_differences.loc[self.time_step - self.lookback + 1 : self.time_step + 1]

        # cast them as tensors
        self.state = torch.Tensor(state)
        self.next_state = torch.Tensor(next_state)

        assert self.state.size() == (self.lookback, 1)
        assert self.next_state.size() == (self.lookback, 1)

        # check the date at index step.
        # If it's the end date, we are at the end of the episode
        self.end = self.price_history[self.date_column].at[self.time_step] == self.end_date

        # increment timestep
        self.time_step += 1


    def add_profit(self, profit):
        self.episode_profit += profit

    def add_loss(self, loss):
        self.episode_losses.append(loss)

    def profit_and_reward(self, action: int, num):

        self.action = action
        self.num_t = num

        action_value = self.action_space[action]

        profit = compute_profit(num_t=num,
                                action_value=action_value,
                                price=self.price,
                                prev_price=self.prev_price)

        reward = compute_reward(num_t=num,
                                action_value=action_value,
                                price=self.price,
                                prev_price=self.prev_price,
                                init_price=self.init_price)

        self.reward = reward

        self.episode_rewards.append(reward)
        self.episode_profit += profit

        return profit, reward


    def update_replay_memory(self):
        self.replay_memory.update(
            (self.state, self.action, self.reward, self.next_state)
        )

    def on_episode_end(self):
        avg_loss = sum(self.episode_losses) / len(self.episode_losses)
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        return avg_loss, avg_reward, self.episode_profit

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def update(self, transition: Tuple[Tensor, int, float, Tensor]):
        """
        Saves a transition
        :param transition: (state, action_index, reward, next_state)
        :return:
        """
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def make_env(index: str, symbol: str, task:str, mem_cap: int, lookback:int):
    """
    :param index: gspc, hsi, etc.
    :param symbol: Ticker
    :param mem_cap: Memory Replay Capacity
    :param lookback: how many days we look back for prices
    :return:
    """
    prices = load_prices(index=index, symbol=symbol)
    return FinanceEnvironment(price_history=prices, mem_cap=mem_cap, index=index, task=task, lookback=lookback)
