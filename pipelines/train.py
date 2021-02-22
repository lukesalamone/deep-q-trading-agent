import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from .build_batches import batched
from utils.rewards import batch_reward, batch_profit
from models.models import NumQModel


"""
TODO a lot
"""


def train_numq(model, dataloaders, threshold, gamma, lr, strategy):
    total_profit = 0
    train, valid, test = dataloaders

    # Initialize dataset
    for i, (states, next_states, price, prev_price) in enumerate(train):
        # See if trade confidence is greater than threshold
        q = model(states)

        q_buy = q[:,model.BUY]
        q_hold = q[:,model.HOLD]
        q_sell = q[:,model.SELL]
        trade_confidence = torch.abs(q_buy - q_sell) / (q_buy + q_hold + q_sell)
        if trade_confidence < threshold:
            pass
        else:
            # Get actions for each state (SELL = -1, HOLD = 0, BUY = 1)
            actions_i = np.argmax(q)
            actions = 1 - actions_i

            # Calculate reward - should give 1 value
            total_profit += batch_profit(actions, states)
            rewards = batch_reward(actions, states)

            # Compute q values for next state
            q_next = model(next_states)
            next_actions_i = np.argmax(q_next)

            # Update model given rewards
            for i in range(states.shape[0]):
                q_s_a = q[i][actions_i[i]]
                q_next_s_next_a = q_next[i][next_actions_i[i]]

                expected_ = rewards[i] + gamma * q_next_s_next_a - q_s_a

                

    return 0


if __name__ == '__main__':
    lr = 1e-4
    model = NumQModel()
    threshold = 0.01
    gamma = 0.85
    strategy = 'HOLD'
    dataloaders = batched(dataset='gspc', batch_size=64)

    train_numq(model, dataloaders, threshold, gamma, lr, strategy)
