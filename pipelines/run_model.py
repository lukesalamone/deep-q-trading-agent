import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import optim, Tensor
import yaml
from itertools import count

from .finance_environment import make_env, ReplayMemory, _reward, _profit
from models.models import *
from 

# Get all config values and hyperparameters
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


# Select an action given model and state
# Returns action index
def select_action(model: DQN, state: Tensor, strategy: int=config["STRATEGY"],
                  strategy_num: float=config["STRATEGY_NUM"], use_strategy=False,
                  only_use_strategy=False):
    # Get q values for this state
    with torch.no_grad():
        q, num = model.policy_net(state)

    # Reduce unnecessary dimension
    q = q.squeeze().detach().numpy()
    num = num.squeeze().detach().numpy()

    action_index = np.argmax(q)
    num = num[action_index]
    num = config["SHARE_TRADE_LIMIT"] * num

    # Use predefined confidence if confidence is too low, indicating a confused market
    confidence = (np.abs(q[model.BUY] - q[model.SELL]) / np.sum(q))
    if only_use_strategy:
        action_index = strategy
        num = strategy_num
    elif use_strategy and confidence < config["THRESHOLD"]:
        action_index = strategy

    # If confidence is high enough, return the action of the highest q value
    return action_index, list(q), num