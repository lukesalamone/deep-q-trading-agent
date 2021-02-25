import numpy as np
from torch import Tensor

def batch_rewards(num_ts:Tensor, action_values:Tensor, prices:Tensor, prev_prices:Tensor, init_prices:Tensor) -> Tensor:
    """
    :param num_ts:
    :param action_values: BUY, HOLD, SELL = [1, 0,-1]
    :param prices:
    :param prev_prices:
    :param init_prices:
    :return:
    """
    rewards = 1 + action_values * (prices - prev_prices) / prev_prices
    rewards = rewards * num_ts * prices / init_prices
    return rewards

def compute_reward(num_t: float, action_value: int, price: float, prev_price: float, init_price: float) -> float:
    """
    :param num_t:
    :param action_value:
    :param price:
    :param prev_price:
    :param init_price:
    :return:
    """
    reward = 1 + action_value * (price - prev_price) / prev_price
    reward = reward * num_t * price / init_price
    return reward

def batch_profits(num_t:np.array, action_values:np.array, prices:np.array, prev_prices:np.array) -> np.array:
    """
    :param num_t:
    :param action_values: BUY, HOLD, SELL = [1, 0,-1]
    :param prices:
    :param prev_prices:
    :return:
    """
    return num_t * action_values * (prices - prev_prices) / prev_prices


def compute_profit(num_t: float, action_value: int, price: float, prev_price: float) -> float:
    """
    :param num_t:
    :param action_value:
    :param price:
    :param prev_price:
    :return:
    """
    return num_t * action_value * (price - prev_price) / prev_price
