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


def train_2(model, dataloader, threshold, batch_size, gamma, lr, strategy):
    losses = []
    for i, (states, next_states) in enumerate(dataloader['train']):
        # Get q values on states batch
        q, r_num = model.policy_net(states).detatch().numpy()
        q_next, r_num_next = model.target_net(states)

        # Branch based on if threshold is high enough
        # Same as above...

        # Compute actions
        action_indices = p.argmax(q)
        actions = 1 - actions_indices

        # Compute rewards
        num = L * r_num[action_indices]
        rewards = batch_reward(actions, num, states)

        # Compute new Q values given q values of the next best action
        # TODO - only for non - terminal states
        updated_q = rewards + gamma * q_next.max()

        # Get new expected Q values for taken actions given new_q
        expected_q = torch.tensor(q).gather(updated_q, action_indices)

        # Compute loss given expected q values
        loss = F.smooth_l1_loos(q, expected_q)
        losses.append(loss.item())

        # Fit model over entire batch/episode
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

if __name__ == '__main__':
    lr = 1e-4
    model = NumQModel()
    threshold = 0.01
    gamma = 0.85
    strategy = 'HOLD'
    dataloaders = batched(dataset='gspc', batch_size=64)

    train_numq(model, dataloaders, threshold, gamma, lr, strategy)
