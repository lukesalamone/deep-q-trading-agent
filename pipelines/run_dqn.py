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
def select_action(model: DQN, state: Tensor, t:int, strategy: int=config["STRATEGY"], strategy_num: float=config["STRATEGY_NUM"], use_exploration=False, use_strategy=False, only_use_strategy=False):
    # Get q values for this state
    with torch.no_grad():
        q, num = model.policy_net(state)

    # Reduce unnecessary dimension
    q = q.squeeze()
    num = num.squeeze()

    if t%500==0:
        print("\tACT: ", q.detach().numpy())
        print("\tNUM: ", num.detach().numpy())
        print()

    # Check (q.shape num.shape should both be (3,) respectively) here
    assert q.shape == (3,)
    # assert num.shape == (3, )

    # Use predefined confidence if confidence is too low, indicating a confused market
    confidence = (torch.abs(q[model.BUY] - q[model.SELL]) / torch.sum(q)).item()
    if use_strategy and confidence < config["THRESHOLD"] or only_use_strategy:
        action_index = strategy
        num = strategy_num
    else:
        action_index = torch.argmax(q).item()

        # Multiply num by trading limit to get actual share trade volume given model method
        if model.method == NUMDREG_ID:
            num = config["SHARE_TRADE_LIMIT"] * num.item()
        else:
            num = config["SHARE_TRADE_LIMIT"] * num[action_index].item()

    # Generate random action and num using epsilon exploration
    if use_exploration:
        if random.random() < config["EPSILON"]:
            action_index = random.randint(0,  2)

        if random.random() < config["EPSILON"]:
            num = random.uniform(0, 1)

    # If confidence is high enough, return the action of the highest q value
    return action_index, num


# Update policy net using a batch from memory
def optimize_model(model: DQN, memory: ReplayMemory):
    # Skip if memory length is not at least batch size
    if len(memory) < config["BATCH_SIZE"]:
        return None

    # Sample a batch from memory
    # (state, action_index, next_state, reward)
    batch = list(zip(*memory.sample(batch_size=config["BATCH_SIZE"])))

    # Get batch of states, actions, and rewards
    # (each item in batch is a tuple of tensors so stack puts them together)
    state_batch = torch.stack(batch[0])
    action_batch = torch.unsqueeze(torch.tensor(batch[1]), dim=1)
    reward_batch = torch.unsqueeze(torch.tensor(batch[2]), dim=1)
    next_state_batch = torch.stack(batch[3])

    # Check shape is (BATCH_SIZE, 200), (BATCH_SIZE, 1), (BATCH_SIZE, 200), (BATCH_SIZE, 1) respectively
    assert state_batch.shape == (config["BATCH_SIZE"], 200)
    assert action_batch.shape == (config["BATCH_SIZE"], 1)
    assert reward_batch.shape == (config["BATCH_SIZE"], 1)
    assert next_state_batch.shape == (config["BATCH_SIZE"], 200)

    # TODO handle terminal state in build_episode and then here?
    # TODO check size of all outputs
    # TODO masking?
    # TODO detach target outputs?
    # TODO need gradient clipping?
    #  RESPONSE: I think we smooth l1 loss takes care of that
    #  https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/

    # Optimize model given specified method
    if model.method == NUMQ:
        loss = optimize_numq(model=model, state_batch=state_batch, action_batch=action_batch, reward_batch=reward_batch,
                             next_state_batch=next_state_batch)
        return (loss,)
    elif model.method == NUMDREG_AD or model.method == NUMDREG_ID:
        # Optimize on step 1
        model.policy_net.set_step(1)
        model.target_net.set_step(1)
        act_loss = optimize_numdreg(model=model, state_batch=state_batch, action_batch=action_batch,
                                    reward_batch=reward_batch, next_state_batch=next_state_batch)

        # Optimize on step 2
        model.policy_net.set_step(2)
        model.target_net.set_step(2)
        num_loss = optimize_numdreg(model=model, state_batch=state_batch, action_batch=action_batch,
                                    reward_batch=reward_batch, next_state_batch=next_state_batch)

        # End to end...
        model.policy_net.set_step(3)
        model.target_net.set_step(3)
        num_loss = optimize_numdreg(model=model, state_batch=state_batch, action_batch=action_batch,
                                    reward_batch=reward_batch, next_state_batch=next_state_batch)

        return (act_loss, num_loss)


def optimize_numq(model, state_batch, action_batch, reward_batch, next_state_batch):
    # Initialize optimizer
    optimizer = optim.Adam(model.policy_net.parameters(), lr=config["LR"])

    # Get q values from policy net (track gradients only for policy net)
    q_batch, num_batch = model.policy_net(state_batch)

    # Get q values from target net with next states
    next_q_batch, next_num_batch = model.target_net(next_state_batch)
    # Get max q values for next state
    next_max_q_batch, next_max_q_i_batch = next_q_batch.detach().max(dim=1)

    # Compute the expected Q values...
    expected_q_batch = q_batch.clone().detach()

    # Fill q values from q batch from index of the taken action to the updated q value using the reward and next max q value
    for i in range(expected_q_batch.shape[0]):
        expected_q_batch[i, action_batch[i]] = reward_batch[i] + (config["GAMMA"] * next_max_q_batch[i])

    # Loss is the difference between the q values from the policy net and expected q values from the target net
    loss = F.smooth_l1_loss(expected_q_batch, q_batch)

    # Clear gradients and update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def optimize_numdreg(model, state_batch, action_batch, reward_batch, next_state_batch):
    # Initialize optimizer
    optimizer = optim.Adam(model.policy_net.parameters(), lr=config["LR"])

    # Get q values from policy and target net from state batch (track gradients only for policy net)
    q_batch, num_batch = model.policy_net(state_batch)
    next_q_batch, next_num_batch = model.target_net(state_batch)

    # Loss is the difference between the q values from the policy net and expected q values from the target net

    # Compute the expected Q values and act loss
    expected_q_batch = reward_batch + (config["GAMMA"] * next_q_batch)
    act_loss = F.smooth_l1_loss(q_batch, expected_q_batch)

    # Compute the expected num values and num loss
    expected_num_batch = reward_batch + (config["GAMMA"] * next_num_batch)
    num_loss = F.smooth_l1_loss(num_batch, expected_num_batch)

    # Clear gradients and update model parameters
    optimizer.zero_grad()

    # Set loss given training step
    if model.policy_net.step == 1:
        loss = act_loss
    elif model.policy_net.step == 2:
        loss = num_loss
    elif model.policy_net.step == 3:
        loss = act_loss + num_loss

    # Update model parameters
    loss.backward()
    optimizer.step()

    return loss.item()


# Train model on given training data
def train(model: DQN, index: str, symbol: str, dataset: str,
          episodes: int=config["EPISODES"], strategy: int=config["STRATEGY"]):
    print(f"Training model on {symbol} from {index} with the {dataset} set...")

    optim_steps = 0
    losses = []
    rewards = []
    total_profits = []
    val_rewards = []
    val_total_profits = []

    # initialize env
    env = make_env(index=index, symbol=symbol, dataset=dataset)

    # Run for the defined number of episodes
    for e in range(episodes):
        # start episode or reset what needs to be reset in the env
        env.start_episode(start_with_padding=False)
        actions = [0,0,0]
        optimal_actions = [0,0,0]

        # iterate until done
        for i in count():
            state, done = env.step()

            action_index, num = select_action(model=model, state=state, t=i, use_exploration=True, use_strategy=False)
            # action_index, num = select_action(model=model, state=state, strategy=strategy, t=i)

            actions[action_index] += 1

            # Compute profit, reward given action_index and num
            _, _, optimal = env.compute_profit_and_reward(action_index=action_index, num=num)

            optimal_actions[optimal] += 1

            # DEBUG
            #if i > -1:
            #    env.print_values()

            # Update memory buffer to include observed transition
            env.update_replay_memory()

            # Update model and increment optimization steps
            loss = optimize_model(model=model, memory=env.replay_memory)
            env.add_loss(loss)

            # Update step
            optim_steps += 1

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

        val_rewards.append(sum(e_val_rewards)/len(e_val_rewards))
        val_total_profits.append(val_total_profit)

        # Update policy net with target net
        if e % config["EPISODES_PER_TARGET_UPDATE"] == 0:
            # TODO NEED A TAU
            model.transfer_weights()

        print(f'actions taken: {actions[0]} buys, {actions[1]} holds, {actions[2]} sells')
        print(f'optimal actions: {optimal_actions[0]} buys, {optimal_actions[2]} sells')

        # Print episode training update
        print("Episode: {} Complete".format(e + 1))
        print("Train: avg_reward={}, total_profit={}, avg_loss={}".format(avg_reward, e_profit, avg_loss))
        print("Valid: avg_reward={}, total_profit={}\n".format(val_rewards[-1], val_total_profit))

    print("Training complete")

    return model, losses, rewards, val_rewards, total_profits, val_total_profits

# Evaluate model on validation or test set and return profits
# Returns a list of profits and total profit
# NOTE only use strategy is if we want to compare against a baseline (buy and hold)
def evaluate(model: DQN, index:str, symbol:str, dataset: str, strategy: int = config["STRATEGY"],
             strategy_num: float = config["STRATEGY_NUM"], use_strategy: bool = False, only_use_strategy: bool = False):
    # TODO: Should strategy be None for training?

    print(f"Evaluating model on {symbol} from {index} with the {dataset} set...")

    # initialize env
    rewards = []
    profits = []
    running_profits = [0]
    env = make_env(index=index, symbol=symbol, dataset=dataset)
    env.start_episode()

    # Look at each time step in the evaluation data
    for i in count():
        state, done = env.step()

        # Select action
        action_index, num = select_action(model=model, state=state, strategy=strategy, strategy_num=strategy_num, 
                                                use_exploration=False, use_strategy=use_strategy, only_use_strategy=only_use_strategy, t=i)

        # Compute profit, reward given action_index and num
        profit, reward, _ = env.compute_profit_and_reward(action_index=action_index, num=num)

        # Add profits to list
        profits.append(profit)
        rewards.append(reward)
        running_profits.append(env.episode_profit)

        if done:
            break

    total_profit = env.episode_profit
    # Return list of profits, running total profits, and total profit
    return rewards, profits, running_profits, total_profit
