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
LR = 0.0001
BATCH_SIZE = 64
GAMMA = 0.05
THRESHOLD = 0.2

# Select an action given model and state
# Returns action index
def select_action(model, state, strategy):
    # Get q values for this state
    with torch.no_grad():
        q, num = model.policy_net(state)
    
    # Reduce unnecessary dimension
    q = q.squeeze()
    num = num.squeeze()
    # TODO check (q.shape num.shape should be (3,) (1,) respectively) here
    
    # Use predefined confidence if confidence is too low, indicating a confused market
    confidence = (torch.abs(q[model.BUY] - q[model.SELL]) / torch.sum(q)).item()
    if confidence < THRESHOLD:
        # TODO use defined strategy (hold for now)
        return model.HOLD, num.item()
    
    # If confidence is high enough, return the action of the highest q value
    return torch.argmax(q).item(), num.item()

# Update policy net using a batch from memory
def optimize_model(model, memory):
    # Skip if memory length is not at least batch size
    if len(memory) < BATCH_SIZE:
        return

    # Initialize optimizer
    optimizer = optim.Adam(model.policy_net.params(),lr=LR)

    # Sample a batch from memory
    # TODO currently using last batch_size transistion, but need to do randomly so it uses action replay
    batch = list(zip(*memory[-BATCH_SIZE]))

    # Get batch of states, actions, and rewards
    # (each item in batch is a tuple of tensors so stack puts them togethor)
    # TODO check if states shape, actions shape, rewards shape is (BATCH_SIZE, 200), (BATCH_SIZE, 1), (BATCH_SIZE, 1) respectively
    state_batch = torch.stack(batch[0])
    action_batch = torch.stack(batch[1])
    reward_batch = torch.stack(batch[2])

    # Get q values from policy net from state batch
    # (we keep track of gradients for this model)
    q_batch, num_batch = model.policy_net(state_batch)

    # Get q values from target net from next states
    # (we do NOT keep track of gradients for this model)
    # TODO remove terminal state
    next_q_batch, next_num_batch = model.target_net(state_batch)

    # TODO check size of all outputs

    # Compute the expected Q values
    expected_q_batch = reward_batch + (GAMMA * next_q_batch)

    # Loss is the difference between the q values from the policy net and expected q values from the target net
    loss = F.smooth_l1_loss(q_batch, expected_q)

    # Clear gradients and update model parameters
    optimizer.zero_grad()
    loss.backward()
    # TODO need gradient clipping?
    optimizer.step()

    return loss.item()


def train(model, num_episodes, strategy=None):
    losses = []
    memory = []

    # Run for the defined number of episodes
    for e in num_episodes:
        # TODO need to fgure out what episode_states should be
        episode_states = None

        for state, next_state in zip(episode_states):
            # Select action
            action_index, num = select_action(model, state, strategy)

            # Get action values from action indices (BUY=1, HOLD=0, SELL=-1)
            action_value = model.action_index_to_value(action_index)

            # Get reward given action_value and num
            reward = get_reward(action_value, num, state)

            # Push transition into memory buffer
            # NOTE (using action index not action value)
            # TODO Need to ensure memory does not exceed certain size?
            memory.append((state, action_index, reward, next_state))

            # Update model and add loss to losses
            loss = optimize_model(model, memory)
            losses.append(loss)

        # Update policy net with target net
        model.transfer_weights()
    
    # Return loss values during training
    return losses