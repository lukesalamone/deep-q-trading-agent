import numpy as np
import torch
from .build_batches import batched
from utils.rewards import batch_rewards, batch_profits
from models.models import NumQModel, NumQDRegModel, DQN
import torch.nn.functional as F
from torch.optim import Adam
from torch import optim

"""
TODO fill in DQN details
"""
L = 10

def select_action(model, state, strategy):
    # Get q values for this state
    with torch.no_grad():
        q = model.policy_net(state)
    
    # Decide whether to use pre defined strategy or action of best q values
    #...

    return action


def optimize_model(model, memory, gamma, batch_size):
    optimizer = optim.Adam(model.policy_net.params(),lr=lr)

    # Sample a batch from memory
    batch = # ...

    # Get batch of states, actions, and rewards
    # TODO unpack from batch
    state_batch = 
    action_batch = 
    reward_batch = 

    # Get q values from policy net from state batch
    # (we keep track of gradients for this model)
    q_batch = model.policy_net(state_batch)

    # Get q values from target net from next states
    # (we do NOT keep track of gradients for this model)
    # TODO remove terminal state
    next_q_batch = model.target_net(state_batch)

    # Compute the expected Q values
    expected_q_batch = reward_batch + (gamma * next_q_batch)

    # Loss is the difference between the q values from the policy net
    # and expected q values from the target net
    loss = F.smooth_l1_loss(q_batch, expected_q)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()



def train3(model, num_episodes, batch_size, threshold, gamma, lr, strategy=None):
    losses = []
    memory = []

    # For each episode
    for e in num_episodes:
        states = #...
        for state in states:
            # Select action
            action, num = select_action(model, state, strategy)

            # Get reward
            reward = batch_rewards(action, num, state)

            # Push transition into memory buffer
            memory.append((state, action, reward, next_state))

            # Train from memory given batch size
            optimize_model(model, memory, batch_size)

        # Update policy net with target net
        model.transfer_weights()
    
    # Return loss values during training
    return losses