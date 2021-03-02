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
    reward =  num_t * reward * prev_price / init_price
    return reward


def compute_profit(num_t: float, action_value: int, price: float, prev_price: float) -> float:
    """
    :param num_t:
    :param action_value:
    :param price:
    :param prev_price:
    :return:
    """
    return num_t * action_value * (price - prev_price) / prev_price
