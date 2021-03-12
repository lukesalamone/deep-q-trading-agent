import numpy as np

import torch.nn.functional as F
from torch import optim, Tensor
import yaml
from itertools import count

from .finance_environment import make_env, ReplayMemory, _reward, _profit
from models.models import *

# Get all config values and hyperparameters
with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


# Select an action index given model and state
def select_action(model: DQN, state: Tensor, strategy: int=config["STRATEGY"],
                  strategy_num: float=config["STRATEGY_NUM"], use_strategy=False,
                  only_use_strategy=False):
    
    # Get q values for this state
    with torch.no_grad():
        q, num = model.policy_net(state)

    # Reduce unnecessary dimension
    q = q.squeeze().detach().numpy()
    num = num.squeeze().detach().numpy()

    # Select action and num
    selected_action_index = np.argmax(q)
    if model.method == NUMDREG_ID:
        if model.mode == ACT_MODE:
            num = list(num)
            selected_num = config["SHARE_TRADE_LIMIT"] * num[0]
        else:
            selected_num = config["SHARE_TRADE_LIMIT"] * num
    else:
        num = list(num)
        selected_num = config["SHARE_TRADE_LIMIT"] * num[selected_action_index]

    # Use predefined confidence if confidence is too low, indicating a confused market
    confidence = (np.abs(q[model.BUY] - q[model.SELL]) / np.sum(q))
    if only_use_strategy:
        selected_action_index = strategy
        selected_num = config["SHARE_TRADE_LIMIT"] * strategy_num
    elif use_strategy and confidence < config["THRESHOLD"]:
        selected_action_index = strategy

    # Return q and num as well as selected action and num
    return list(q), selected_action_index, num, selected_num


# Get batches from replay memory
def get_batches(memory: ReplayMemory):
    # Sample a batch from memory (state, action_index, next_state, reward)
    batch = list(zip(*memory.sample(batch_size=config["BATCH_SIZE"])))

    # Get batch of states, actions, and rewards (each item in batch is a tuple of tensors so stack puts them together)
    state_batch = torch.stack(batch[0])
    action_batch = torch.unsqueeze(torch.tensor(batch[1]), dim=1)
    reward_batch = torch.stack(batch[2])
    next_state_batch = torch.stack(batch[3])

    return state_batch, action_batch, reward_batch, next_state_batch

# Update policy net using a batch from memory
def optimize_model(model: DQN(NUMQ), optimizer, memory: ReplayMemory, optim_actions:bool=True):
    # Get transition batches
    state_batch, action_batch, reward_batch, next_state_batch = get_batches(memory)
    
    # Get q values from policy net (track gradients only for policy net)
    q_batch, num_batch = model.policy_net(state_batch)
    # Get q values from target net with next states
    next_q_batch, next_num_batch = model.target_net(next_state_batch)

    # Decide whether we optimize q values of num values
    if optim_actions:
        pred_batch = q_batch
        next_pred_batch = next_q_batch
    else:
        pred_batch = num_batch
        next_pred_batch = next_num_batch

    # Get max q values for next state
    next_max_pred_batch, next_max_pred_i_batch = next_pred_batch.detach().max(dim=1)

    # Compute the expected Q values
    expected_pred_batch = reward_batch + (config["GAMMA"] * torch.unsqueeze(next_max_pred_batch, dim=1))

    # Loss is the difference between the q values from the policy net and expected q values from the target net
    if config["LOSS"] == "SMOOTH_L1_LOSS":
        loss = F.smooth_l1_loss(expected_pred_batch, pred_batch)
    elif config["LOSS"] == "MSE_LOSS":
        loss = F.mse_loss(expected_pred_batch, pred_batch)
    else:
        loss = F.smooth_l1_loss(expected_pred_batch, pred_batch)

    # Clear gradients and update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Return loss value
    return loss.item()

# Train model on given training data
def run_train_loop(model: DQN, index: str, optimizer, symbol: str, dataset: str, episodes: int = config["EPISODES"], 
                   strategy: int = config["STRATEGY"], use_strategy: bool=config["USE_STRATEGY_TRAIN"], path: str=config["STOCK_DATA_PATH"], 
                   splits=config["INDEX_SPLITS"]):

    print(f"Training model on {symbol} from {index} with {dataset} for {episodes} episodes...")

    optim_steps = 0
    
    # Track losses rewards and profits
    losses = []
    rewards = []
    total_profits = []
    val_rewards = []
    val_total_profits = []

    # initialize env
    env = make_env(index=index, symbol=symbol, dataset=dataset, path=path, splits=splits)

    # Run for the defined number of episodes
    for e in range(episodes):
        # Track porportion of actions taken
        actions_taken = [0, 0, 0]

        # start episode or reset what needs to be reset in the env
        env.start_episode(start_with_padding=False)

        # iterate until done
        for i in count():
            # Get state and terminal boolean from environment
            state, done = env.step()

            # Get action index and num shares to trade
            q_values, selected_action_index, num_values, selected_num = select_action(model=model, state=state)

            # Update actions taken
            actions_taken[selected_action_index] += 1

            # Compute profit and reward for all actions
            env.compute_reward_all_actions(action_index=selected_action_index, num=selected_num)

            # Update memory buffer to include observed transition
            env.update_replay_memory()

            # Go to next step if memory not large enough yet
            if len(env.replay_memory) < config["BATCH_SIZE"]:
                continue

            # Update model and increment optimization steps based on model method
            if model.method == NUMDREG_AD or model.method == NUMDREG_ID:
                if model.mode == ACT_MODE:
                    # Optimize actions
                    loss = optimize_model(model=model, optimizer=optimizer, memory=env.replay_memory, optim_actions=True)
                
                elif model.mode == NUM_MODE:
                    # Optimize num
                    loss = optimize_model(model=model, optimizer=optimizer, memory=env.replay_memory, optim_actions=False)

                elif model.mode == FULL_MODE:
                    # Optimize both num and actions
                    act_loss = optimize_model(model=model, optimizer=optimizer, memory=env.replay_memory, optim_actions=True)
                    num_loss = optimize_model(model=model, optimizer=optimizer, memory=env.replay_memory, optim_actions=False)

                    # Add rewards
                    loss = act_loss + num_loss
            else:
                loss = optimize_model(model=model, optimizer=optimizer, memory=env.replay_memory)
            
            env.add_loss(loss)

            # Update step
            optim_steps += 1

            # Soft TAU update
            if config["UPDATE_TYPE"] == "SOFT" and optim_steps % config["STEPS_PER_SOFT_UPDATE"] == 0:
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
        e_val_rewards, _, _, val_total_profit = evaluate(model, index=index, symbol=symbol, dataset='valid', path=path, splits=splits)
        val_rewards.append(sum(e_val_rewards) / len(e_val_rewards))
        val_total_profits.append(val_total_profit)

        # Update policy net with target net
        if config["UPDATE_TYPE"] == "HARD" and optim_steps % config["EPISODES_PER_TARGET_UPDATE"] == 0:
            model.hard_update()

        # Print episode training update
        print("Episode: {} Complete".format(e + 1))
        print("Train: avg_reward={}, total_profit={}, avg_loss={}".format(avg_reward, e_profit, avg_loss))
        print("Valid: avg_reward={}, total_profit={}\n".format(val_rewards[-1], val_total_profit))

    print("Training complete")

    # Return model, losses, and performance metrics on rewards and profits
    return model, losses, rewards, val_rewards, total_profits, val_total_profits


def train(model: DQN, index: str, symbol: str, dataset: str, episodes: int=config["EPISODES"], pretrain:bool = False,
          strategy: int=config["STRATEGY"], path: str=config["STONK_PATH"], splits=config["STONK_INDEX_SPLITS"]):
    # Train NumQ
    if model.method == NUMQ:
        print("Training NumQ...")
        # Initialize optimizer
        optimizer = optim.Adam(model.policy_net.parameters(), lr=config["LR"])
        return run_train_loop(model=model, index=index, optimizer=optimizer, symbol=symbol, dataset=dataset, episodes=episodes,
                              strategy=strategy, path=path, splits=splits)
    
    # Train NumDReg
    elif model.method == NUMDREG_AD or model.method == NUMDREG_ID:
        if model.method == NUMDREG_AD:
            print("Training NumDReg-AD...")
            # Initialize optimizer
            optimizer = optim.Adam(model.policy_net.parameters(), lr=config["LR_NUMDREGAD"])
        elif model.method == NUMDREG_ID:
            print("Training NumDReg-ID...")
            # Initialize optimizer
            optimizer = optim.Adam(model.policy_net.parameters(), lr=config["LR_NUMDREGID"])

        # Train action branch
        model.set_mode(ACT_MODE)
        print("Training Action Branch...")
        act_trained = run_train_loop(model=model, index=index, optimizer=optimizer, symbol=symbol, dataset=dataset, episodes=episodes,
                                     strategy=strategy, path=path, splits=splits)
        if pretrain:
            return act_trained

        # Train num branch
        model.set_mode(NUM_MODE)
        print("Training Number Branch...")
        num_trained = run_train_loop(model=model, index=index, optimizer=optimizer, symbol=symbol, dataset=dataset, episodes=episodes,
                                     strategy=strategy, path=path, splits=splits)

        # Train both branches
        model.set_mode(FULL_MODE)
        print("Training Full...")
        full_trained = run_train_loop(model=model, index=index, optimizer=optimizer, symbol=symbol, dataset=dataset, episodes=episodes,
                                      strategy=strategy, path=path, splits=splits)

        return full_trained
    
    print("Invalid model method")
    return

# Evaluate model on validation or test set and return list of profits and total profits
# NOTE only use strategy is if we want to compare against a baseline (buy and hold)
def evaluate(model: DQN, index: str, symbol: str, dataset: str, strategy: int=config["STRATEGY"], strategy_num: float=config["STRATEGY_NUM"],
             use_strategy: bool=False, only_use_strategy: bool = False, path: str=config["STOCK_DATA_PATH"], splits=config["INDEX_SPLITS"]):

    print(f"Evaluating model on {symbol} from {index} with {dataset}...")

    # Track rewards, profits, and actions taken
    rewards = []
    profits = []
    running_profits = [0]
    actions_taken = [0, 0, 0]

    # initialize env
    env = make_env(index=index, symbol=symbol, dataset=dataset, path=path, splits=splits)
    env.start_episode()

    # Look at each time step in the evaluation data
    for i in count():
        state, done = env.step()

        # Select action
        q_values, selected_action_index, num_values, selected_num = select_action(model=model, state=state, 
                                                                                  strategy=strategy, strategy_num=strategy_num, 
                                                                                  use_strategy=use_strategy, 
                                                                                  only_use_strategy=only_use_strategy)

        # log actions
        actions_taken[selected_action_index] += 1

        # Compute profit, reward given action_index and num
        profit, reward = env.compute_profit_and_reward(action_index=selected_action_index, num=selected_num)

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
