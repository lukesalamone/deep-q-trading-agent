import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from ..utils.rewards import batch_reward, batch_profit

"""
TODO a lot
"""
def train_numq(model, dataloader, threshold, batch_size, gamma, lr, strategy):
    # Set initial profit to 0
    total_profit = 0

    # Initialize model optimizer

    # Initialize dataset
    for i, (states, next_states) in enumerate(dataloader['train']):
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

