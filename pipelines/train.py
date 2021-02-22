import numpy as np
import torch
from .build_batches import batched
from utils.rewards import batch_rewards, batch_profits
from models.models import NumQModel


"""
TODO a lot
"""
L = 10



def train_numq(model, dataloaders, threshold, gamma, lr, strategy=None):
    total_profit = 0
    train, valid, test = dataloaders

    # Initialize dataset
    for i, (states, next_states, prices, prev_prices, init_prices) in enumerate(train):
        q_table, ratios = model(states)

        q_buy = q_table[:,model.BUY]
        q_hold = q_table[:,model.HOLD]
        q_sell = q_table[:,model.SELL]
        confidences = torch.abs(q_buy - q_sell) / (q_buy + q_hold + q_sell)

        if strategy is not None:
            actions = [strategy if c < threshold else torch.argmax(qvals) for c, qvals in zip(confidences, q_table)]
        else:
            # get argmax for each item in batch
            actions = torch.argmax(q_table, dim=1)

        # calculate initial prices p_{t-n}
        num_ts = ratios[:,actions]
        rewards = batch_rewards(num_ts=num_ts, actions=actions, prices=prices,
                               prev_prices=prev_prices, init_prices=init_prices)

        profits = batch_profits(num_t=num_ts, actions=actions, prices=prices,
                                prev_prices=prev_prices)

        buffer = list(zip(states, actions, rewards, next_states))

        #
        # if trade_confidence < threshold:
        #     pass
        # else:
        #     # Get actions for each state (SELL = -1, HOLD = 0, BUY = 1)
        #     actions_i = np.argmax(q)
        #     actions = 1 - actions_i
        #
        #     # Calculate reward - should give 1 value
        #     total_profit += batch_profit(actions, states)
        #     rewards = batch_reward(actions, states)
        #
        #     # Compute q values for next state
        #     q_next = model(next_states)
        #     next_actions_i = np.argmax(q_next)
        #
        #     # Update model given rewards
        #     for i in range(states.shape[0]):
        #         q_s_a = q[i][actions_i[i]]
        #         q_next_s_next_a = q_next[i][next_actions_i[i]]
        #
        #         expected_ = rewards[i] + gamma * q_next_s_next_a - q_s_a
        #
        #

    return 0


if __name__ == '__main__':
    lr = 1e-4
    model = NumQModel()
    threshold = 0.01
    gamma = 0.85
    strategy = 'HOLD'
    dataloaders = batched(dataset='gspc', batch_size=64)

    train_numq(model, dataloaders, threshold, gamma, lr, strategy)
