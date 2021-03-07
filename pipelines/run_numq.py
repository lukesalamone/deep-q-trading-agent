import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import optim, Tensor
import yaml
from itertools import count

from .build_batches import get_episode
from .finance_environment import make_env, ReplayMemory, _reward, _profit
from models.models import *

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


# Update policy net using a batch from memory
def optimize_model(model: DQN(NUMQ), optimizer, memory: ReplayMemory):
    # Skip if memory length is not at least batch size
    if len(memory) < config["BATCH_SIZE"]:
        return None

    # Sample a batch from memory (state, action_index, next_state, reward)
    batch = list(zip(*memory.sample(batch_size=config["BATCH_SIZE"])))

    # Get batch of states, actions, and rewards (each item in batch is a tuple of tensors so stack puts them together)
    state_batch = torch.stack(batch[0])
    # action_batch = torch.unsqueeze(torch.tensor(batch[1]), dim=1)
    reward_batch = torch.stack(batch[2])
    next_state_batch = torch.stack(batch[3])

    # Optimize model given specified method
    if model.method == NUMQ:
        loss = optimize_numq(model=model, optimizer=optimizer, state_batch=state_batch,
                             reward_batch=reward_batch, next_state_batch=next_state_batch)
    return loss

def optimize_numq(model, optimizer, state_batch, reward_batch, next_state_batch):
    # Get q values from policy net (track gradients only for policy net)
    q_batch, num_batch = model.policy_net(state_batch)

    # Get q values from target net with next states
    next_q_batch, next_num_batch = model.target_net(next_state_batch)

    # Get max q values for next state
    next_max_q_batch, next_max_q_i_batch = next_q_batch.detach().max(dim=1)

    # Compute the expected Q values...
    expected_q_batch = reward_batch + (config["GAMMA"] * torch.unsqueeze(next_max_q_batch, dim=1))

    # Loss is the difference between the q values from the policy net and expected q values from the target net
    if config["LOSS"] == "SMOOTH_L1_LOSS":
        loss = F.smooth_l1_loss(expected_q_batch, q_batch)
    elif config["LOSS"] == "MSE_LOSS":
        loss = F.mse_loss(expected_q_batch, q_batch)
    else:
        loss = F.smooth_l1_loss(expected_q_batch, q_batch)

    # Clear gradients and update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Return loss value
    return loss.item()

# Train model on given training data
def train(model: DQN, index: str, symbol: str, dataset: str,
          episodes: int = config["EPISODES"], strategy: int = config["STRATEGY"]):

    print(f"Training model on {symbol} from {index} with the {dataset} set...")

    optim_steps = 0
    epsilon = config["EPSILON"]
    losses = []
    rewards = []
    total_profits = []
    val_rewards = []
    val_total_profits = []

    # Initialize optimizer
    optimizer = optim.Adam(model.policy_net.parameters(), lr=config["LR"])

    # initialize env
    env = make_env(index=index, symbol=symbol, dataset=dataset)

    # Run for the defined number of episodes
    for e in range(episodes):
        actions_taken = [0, 0, 0]
        # start episode or reset what needs to be reset in the env
        env.start_episode(start_with_padding=False)

        # iterate until done
        for i in count():
            state, done = env.step()

            # Get action index and num shares to trade
            # action_index, num = select_action(model=model, state=state, epsilon=epsilon, t=i, use_strategy=False)
            action_index, q_action, num = select_action(model=model, state=state)

            actions_taken[action_index] += 1
            # Compute profit, reward given action_index and num
            # env.compute_profit_and_reward(action_index=action_index, num=num)
            # compute all rewards
            env.compute_reward_all_actions(action_index=action_index, num=num)

            # Update memory buffer to include observed transition
            env.update_replay_memory()

            # Update model and increment optimization steps
            loss = optimize_model(model=model, optimizer=optimizer, memory=env.replay_memory)
            env.add_loss(loss)

            # Update step
            optim_steps += 1

            # Soft TAU update
            if config["UPDATE_TYPE"] == "SOFT":
                if optim_steps % config["STEPS_PER_SOFT_UPDATE"] == 0:
                    model.soft_update(tau=config["TAU"])

            # Break loop if at terminal state
            if done:
                break

        # Update training performance metrics
        avg_loss, avg_reward, e_profit = env.on_episode_end()

        losses.append(avg_loss)
        rewards.append(avg_reward)
        total_profits.append(e_profit)

        # Update validation performance metrics
        e_val_rewards, _, _, val_total_profit = evaluate(model, index=index, symbol=symbol, dataset='valid')

        val_rewards.append(sum(e_val_rewards) / len(e_val_rewards))
        val_total_profits.append(val_total_profit)

        # Update policy net with target net
        if config["UPDATE_TYPE"] == "HARD":
            if optim_steps % config["EPISODES_PER_TARGET_UPDATE"] == 0:
                model.hard_update()

        # Print episode training update
        print("Episode: {} Complete".format(e + 1))
        print("Train: avg_reward={}, total_profit={}, avg_loss={}".format(avg_reward, e_profit, avg_loss))
        print("Valid: avg_reward={}, total_profit={}\n".format(val_rewards[-1], val_total_profit))

    print("Training complete")

    return model, losses, rewards, val_rewards, total_profits, val_total_profits


# Evaluate model on validation or test set and return profits
# Returns a list of profits and total profit
# NOTE only use strategy is if we want to compare against a baseline (buy and hold)
def evaluate(model: DQN, index: str, symbol: str, dataset: str,
             strategy: int = config["STRATEGY"], strategy_num: float = config["STRATEGY_NUM"],
             use_strategy: bool = False, only_use_strategy: bool = False):

    # TODO: Should strategy be None for training?

    print(f"Evaluating model on {symbol} from {index} with the {dataset} set...")

    # initialize env
    rewards = []
    profits = []
    running_profits = [0]
    actions_taken = [0, 0, 0]
    env = make_env(index=index, symbol=symbol, dataset=dataset)
    env.start_episode()

    # Look at each time step in the evaluation data
    for i in count():
        state, done = env.step()

        # Select action
        action_index, _, num = select_action(model=model, state=state, use_strategy=use_strategy, only_use_strategy=only_use_strategy)

        # log actions
        actions_taken[action_index] += 1

        # Compute profit, reward given action_index and num
        profit, reward = env.compute_profit_and_reward(action_index=action_index, num=num)

        # Add profits to list
        profits.append(profit)
        rewards.append(reward)
        running_profits.append(env.episode_profit)

        # Break loop if at terminal state
        if done:
            break

    # Compute total profit
    total_profit = env.episode_profit

    # Print action proportions
    print("Actions taken : {}\n".format(actions_taken))

    # Return list of profits, running total profits, and total profit
    return rewards, profits, running_profits, total_profit
