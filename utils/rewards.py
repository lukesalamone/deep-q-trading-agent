import numpy as np

# TODO batch reward function
def batch_reward(prev_states:np.array, states:np.array, actions:np.array) -> np.array:
    pass

def batch_profit(num_t:np.array, actions:np.array, prices:np.array, prev_prices:np.array) -> np.array:
    return num_t * actions * (prices - prev_prices) / prev_prices

# TODO convert from raw prices
def profit(action:int, prices:np.array, t:int, num_t:float=1) -> float:
    """
    :param action: Action in A = {1, 0, -1}
    :param prices: array of daily trade closing prices
    :param t: time step
    :param num_t: number of shares
    :return: daily profit
    """
    # TODO: Check if num_t is a float (eq. 16)
    return num_t * action * ( prices[t] - prices[t-1] ) / prices[t-1]


def reward(action:int, prices:np.array, t:int, num_t:float=1, n:int=200) -> float:
    """
    :param action: Action in A = {1, 0, -1}
    :param prices: array of daily trade closing prices
    :param t: time step
    :param num_t: number of shares
    :parma n: period of time
    :return: reward over period n
    :return:
    """
    return ( 1 + profit(action, prices, t, num_t) ) * prices[t-1] / prices[t-n]


def total_profits(profits:np.array) -> float:
    """
    :param prices: array of daily trade closing prices
    :return:
    """
    return np.sum(profits)
