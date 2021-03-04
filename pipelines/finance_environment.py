from collections import deque
from typing import Tuple
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
        self.price_history = price_history
        self.lookback = config["LOOKBACK"]

        # BUY, HOLD, SELL, = (1, 0, -1
        self.action_space = (1, 0, -1)

        # pad using start date and set time_step at lookback + start_index
        self.profits = []
        self.replay_memory = ReplayMemory(capacity=config["MEMORY_CAPACITY"])

        self.date_column, self.price_column = self.price_history.columns
        self.init_prices()

    def init_prices(self):
        # we get the start index based on start_date.
        self.start_index = self.price_history[self.date_column].get_loc[self.start_date]

        # if lookback > t we are in trouble,
        # so we pad this dataframe with rows identical to the first row
        padding = self.price_history.at[0] * self.lookback
        self.price_history = pd.concat([padding, self.price_history], axis=1, ignore_index=True)
        self.historical_price_differences = self.price_history[self.price_column].diff(1)

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
        state = self.historical_price_differences.loc[self.time_step - self.lookback : self.time_step]
        self.state = torch.Tensor(state)
        assert self.state.size() == self.lookback

        # check the date at index step.
        # If it's the end date, we are at the end of the episode
        # end = self.price_history[self.date_column].at[self.time_step] == self.end_date
        end = self.price_history[self.date_column].at[self.time_step] >= self.end_date

        if not end:
            # get the next state
            next_state = self.historical_price_differences.loc[self.time_step - self.lookback + 1 : self.time_step + 1]
            self.next_state = torch.Tensor(next_state)
            assert self.next_state.size() == self.lookback
        else:
            self.next_state = None

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
        if self.next_state:
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
