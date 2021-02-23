import numpy as np
import torch
from .build_batches import batched
from utils.rewards import batch_rewards, batch_profits
from models.models import NumQModel, NumQDRegModel, DQN
import torch.nn.functional as F
from torch.optim import Adam
from torch import optim

"""
TODO a lot
"""
L = 10



def train_numq(model, dataloaders, threshold, gamma, lr, strategy=None):
    total_profit = 0
    optimizer = optim.Adam(model.policy_net.parameters(), lr=lr)
    train, valid, test = dataloaders
    losses = []
    # Initialize dataset
    for i, (states, next_states, prices, prev_prices, init_prices) in enumerate(train):
        # q_table is (batch_size, 3)
        # ratios is (batch_size, 3)
        q_table, ratios = model.policy_net(states)
        q_next, ratios_next = model.target_net(next_states)
        q_table, ratios = q_table.detach(), ratios.detach()

        # we slice by column to get buy, hold, sell values (batch_size, 1)
        q_buy = q_table[:,model.BUY]
        q_sell = q_table[:,model.SELL]

        # (64,)
        confidences = torch.abs(q_buy - q_sell) / torch.sum(input=q_table, dim=1)
        best_q = torch.argmax(q_table, dim=1)

        if strategy is not None:
            actions = torch.tensor([strategy if c < threshold else best_action for c, best_action in zip(confidences, best_q)])
        else:
            # use best action for each sample
            actions = best_q

        # calculate initial prices p_{t-n}
        num_ts = ratios[:,actions]
        rewards = batch_rewards(num_ts=num_ts, actions=actions, prices=prices,
                               prev_prices=prev_prices, init_prices=init_prices)

        total_profit += batch_profits(num_t=num_ts, actions=actions, prices=prices,
                                prev_prices=prev_prices)

        updated_q = rewards + gamma * q_next.max()
        # TODO this isn't right...
        # updated_q should be (64,3) not (64, 64)

        expected_q = torch.gather(input=updated_q, dim=0, index=actions)

        loss = F.smooth_l1_loss(q_table, expected_q)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return 0


def train_numq2(model, dataloaders, threshold, gamma, lr, strategy=None):
    losses = []
    optimizer = optim.Adam(model.params(),lr=lr)
    train, valid, test = dataloaders

    for i, (states, next_states, prices, prev_prices, init_prices) in enumerate(train):
        # Get q values on states batch
        q, r_num = model.policy_net(states).detatch().numpy()
        q_next, r_num_next = model.target_net(next_states) # next_states??

        # Branch based on if threshold is high enough
        # Same as above...

        # Compute actions
        action_indices = np.argmax(q, axis=1)
        actions = 1 - action_indices

        # Compute rewards
        num = L * r_num[action_indices]
        rewards = batch_rewards(actions, num, states)

        # Compute new Q values given q values of the next best action
        # TODO - only for non - terminal states
        updated_q = rewards + gamma * q_next.max()

        # Get new expected Q values for taken actions given new_q
        expected_q = torch.tensor(q).gather(updated_q, action_indices)

        # TODO double check gather()

        # Compute loss given expected q values
        loss = F.smooth_l1_loss(q, expected_q)
        losses.append(loss.item())

        # Fit model over entire batch/episode
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update policy net with target net
        model.transfer_weights()
        # TODO potentially use tau param
        

if __name__ == '__main__':
    lr = 1e-4
    model = NumQModel()
    threshold = 0.01
    gamma = 0.85
    strategy = 'HOLD'
    dataloaders = batched(dataset='gspc', batch_size=64)

    train_numq(model, dataloaders, threshold, gamma, lr, strategy)
