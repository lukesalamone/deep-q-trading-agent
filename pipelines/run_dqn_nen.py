from pipelines.transfer_learning import gather_groups
from pipelines.run_dqn import train, evaluate
from models.models import *
from typing import List, Dict
import yaml
import json
import os
import matplotlib.pyplot as plt
from copy import deepcopy

# TODO:
#   1. MAKE SURE THIS WORKS FOR NUMDREG models
#   2. PRETRAINING on groups, for NUMDREG: we use NUMQ. only once a group is selected do we use steps 2 and 3

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

MODEL_METHODS = ['numq', 'numdreg_ad', 'numdreg_id']
# METHODS = ['numq', 'numdreg', 'numdreg']

def load_weights(model: DQN, IN_PATH):
    model.policy_net.load_state_dict(torch.load(IN_PATH))
    model.hard_update()
    return model

def save_weights(model: DQN, OUT_PATH):
    torch.save(model.target_net.state_dict(), OUT_PATH)
    return

def log_experiment(experiment:Dict, filename:str, path:str=config["EXPERIMENT_LOGS_PATH"]):
    outpath = os.path.join(path, f"{filename}.json")
    with open(outpath, 'w') as outfile:
        json.dump(experiment, outfile, indent=4)

def load_experiment(filename:str, path:str=config["EXPERIMENT_LOGS_PATH"]):
    inpath = os.path.join(path, f"{filename}.json")
    with open(inpath, 'r') as infile:
        data = json.load(infile)
    return data

def pretrain_on_group(model: DQN, groups: Dict, index:str, method: str, group:str, train_set:str='train',
                      eval_set:str='valid', load:bool=True):

    group_name = f"{method} - {group}"
    dirname = MODEL_METHODS[model.method]

    if load:
        try:
            pretrained_weights = f'weights/{dirname}/{index}/{method}/{group}/{dirname}_pretrain.pt'
            model = load_weights(model=model, IN_PATH=pretrained_weights)

            previous_experiment = f"{dirname}.{index}.{method}.{group}"
            experiment_log = load_experiment(filename=previous_experiment)

            print(f" ----- STEP 1: START GROUP TRAINING => LOADED PREVIOUS WEIGHTS ---- ")
            print(f"index: {index}, group: {group_name} ...")

            return model, experiment_log
        except:
            pass

    experiment_log = {
        'name': group_name,
        'index': index,
        'model-method': model.method,
        'dirname': dirname,
        'method': method,
        'group': group,
        'train set': train_set,
        'eval set': eval_set,
        'episodes on components': config["EPISODES_COMPONENT_STOCKS"],
        'total profits': 0
    }

    print(f" ----- STEP 1: START GROUP TRAINING ---- ")
    print(f"index: {index}, group: {group_name} ...")
    component_stocks = len(groups[index][method][group])

    for i, symbol in enumerate(groups[index][method][group]):
        print(f"progress: stock {i}/{component_stocks} ...")
        model, losses, rewards, val_rewards, profits, val_profits = train(model=model,
                                                                          index=index,
                                                                          symbol=symbol,
                                                                          episodes=config["EPISODES_COMPONENT_STOCKS"],
                                                                          dataset=train_set,
                                                                          pretrain=True,
                                                                          path=config['STONK_PATH'],
                                                                          splits=config['STONK_INDEX_SPLITS'])

        eval_rewards, eval_profits, eval_running_profits, eval_total_profits = evaluate(model=model,
                                                                                        index=index,
                                                                                        symbol=symbol,
                                                                                        dataset=eval_set,
                                                                                        path=config['STONK_PATH'],
                                                                                        splits=config['STONK_INDEX_SPLITS'])

        experiment_log[symbol] = {
            'train': {
                'losses': losses,
                'rewards':rewards,
                'val rewards': val_rewards,
                'profits': profits,
                'val profits': val_profits
            },
            'component_eval': {
                'rewards': eval_rewards,
                'profits': eval_profits,
                'running profits': eval_running_profits,
                'total profits': eval_total_profits
            }
        }
        # break

        print(f" ----- STEP 2: EVALUATE GROUP MODEL ON INDEX ---- ")
        print(f"index: {index}, group: {group_name} ...")
        idx_eval_rewards, idx_eval_profits, idx_eval_running_profits, idx_eval_total_profits = evaluate(model=model,
                                                                                                        index=index,
                                                                                                        symbol=config["SYMBOLS_DICT"][index],
                                                                                                        dataset=eval_set,
                                                                                                        path=config['STONK_PATH'],
                                                                                                        splits=config['STONK_INDEX_SPLITS'])
    
        experiment_log["eval results on index"] = {
            'rewards': idx_eval_rewards,
            'profits': idx_eval_profits,
            'running profits': idx_eval_running_profits,
            'total profits': idx_eval_total_profits
        }
        experiment_log["total profits"] = idx_eval_total_profits

        # we save the weights for this group
        pretrained_weights = f'weights/{dirname}/{index}/{method}/{group}/{dirname}_pretrain.pt'
        save_weights(model=model, OUT_PATH=pretrained_weights)

        experiment_log["pretrained weights"] = pretrained_weights
        log_experiment(experiment=experiment_log, filename=f"{dirname}.{index}.{method}.{group}")

    return model, experiment_log


def evaluate_groups(model_method:int, groups:Dict, index: str, train_set:str='train', eval_set:str='valid', load:bool=True):
    best_group = {
        'group name': '',
        'profit': float('-inf'),
        'pretrained weights': ''
    }
    results = {
    }
    for method in groups[index]:
        for group in groups[index][method]:
            group_name = f"{method} - {group}"
            model = DQN(method=model_method)
            _, log = pretrain_on_group(model=model, groups=groups, index=index, method=method, group=group, load=load,
                                       train_set=train_set, eval_set=eval_set)
            results[group_name] = deepcopy(log["eval results on index"])
            if log["total profits"] > best_group["profit"]:
                best_group["group name"] = group_name
                best_group["profit"] = log["total profits"]
                best_group["pretrained weights"] = log["pretrained weights"]
            # break

    model = DQN(method=model_method)
    model, _, _, _, _, _ = train(model=model, index=index, symbol=config["SYMBOLS_DICT"][index],
                                 episodes=config["EPISODES_COMPONENT_STOCKS"], dataset=train_set, pretrain=True,
                                 path=config['STONK_PATH'], splits=config['STONK_INDEX_SPLITS'])

    rl_rewards, rl_profits, rl_running_profits, rl_total_profits = evaluate(model,
                                                                            index=index,
                                                                            symbol=config["SYMBOLS_DICT"][index],
                                                                            dataset=eval_set,
                                                                            path=config['STONK_PATH'],
                                                                            splits=config['STONK_INDEX_SPLITS'])

    results['RL'] = {
        'total profits': rl_total_profits,
        'rewards': rl_rewards,
        'profits': rl_profits,
        'running profits': rl_running_profits
    }

    mkt_rewards, mkt_profits, mkt_running_profits, mkt_total_profits = evaluate(model, index=index,
                                                                                symbol=config["SYMBOLS_DICT"][index],
                                                                                dataset=eval_set, strategy=0,
                                                                                strategy_num=1.0, only_use_strategy=True,
                                                                                path=config['STONK_PATH'],
                                                                                splits=config['STONK_INDEX_SPLITS'])

    results['MKT'] = {
        'total profits': mkt_total_profits,
        'rewards': mkt_rewards,
        'profits': mkt_profits,
        'running profits': mkt_running_profits
    }

    dirname = MODEL_METHODS[model_method]
    plots_path = f"{dirname}/{index}"
    plot_group_eval_results(results=results, path=plots_path)

    return best_group, results

def run_dqn_nen_on_index(model_method: int, index: str, symbol: str, train_set: str='train', eval_set: str='valid',
                         load: bool=config["LOAD_PREV_EXPERIMENTS"], path=config["STONK_PATH"], splits=config["STONK_INDEX_SPLITS"]):

    dirname = MODEL_METHODS[model_method]
    groups = gather_groups()
    best_group, results = evaluate_groups(model_method=model_method, groups=groups, index=index, train_set=train_set,
                                          eval_set=eval_set, load=load)

    previous_weights = best_group["pretrained weights"]

    model = DQN(model_method)
    # model = DQN(NUMQ)
    model = load_weights(model=model, IN_PATH=previous_weights)

    print(f"Start Train on {index}, symbol: {symbol} ... ")
    model, losses, rewards, val_rewards, profits, val_profits = train(model=model,
                                                                      index=index,
                                                                      symbol=symbol,
                                                                      episodes=config["EPISODES"],
                                                                      dataset=train_set,
                                                                      pretrained=True,
                                                                      path=path,
                                                                      splits=splits)

    # MKT on training
    print('MKT BUY on Training')
    mkt_train_rewards, mkt_train_profits, mkt_train_running_profits, mkt_train_total_profits = evaluate(model,
                                                                                                        index=index,
                                                                                                        symbol=symbol,
                                                                                                        dataset=train_set,
                                                                                                        strategy=0,
                                                                                                        strategy_num=1.0,
                                                                                                        only_use_strategy=True,
                                                                                                        path=path,
                                                                                                        splits=splits)

    # MKT on validation
    print('MKT BUY on Eval Set')
    mkt_valid_rewards, mkt_valid_profits, mkt_valid_running_profits, mkt_valid_total_profits = evaluate(model,
                                                                                                        index=index,
                                                                                                        symbol=symbol,
                                                                                                        dataset=eval_set,
                                                                                                        strategy=0,
                                                                                                        strategy_num=1.0,
                                                                                                        only_use_strategy=True,
                                                                                                        path=path,
                                                                                                        splits=splits)


    plots_path= f"{dirname}/{index}"
    plot_losses(losses=losses, path=plots_path)
    plot_rewards(rewards=rewards, val_rewards=val_rewards, path=plots_path)
    plot_profits_train(profits=profits, val_profits=val_profits,
                       mkt_train_total_profits=mkt_train_total_profits,
                       mkt_valid_total_profits=mkt_valid_total_profits,
                       path=plots_path)

    print(f"Start Eval on {index}, symbol: {symbol} ... ")
    eval_rewards, eval_profits, eval_running_profits, eval_total_profits = evaluate(model=model,
                                                                                   index=index,
                                                                                   symbol=symbol,
                                                                                   dataset=eval_set,
                                                                                   path=path,
                                                                                   splits=splits)

    # MKT buying and holding 1 share
    mkt_rewards, mkt_profits, mkt_running_profits, mkt_total_profits = evaluate(model,
                                                                                index=index,
                                                                                symbol=symbol,
                                                                                dataset=eval_set,
                                                                                strategy=0,
                                                                                strategy_num=1.0,
                                                                                only_use_strategy=True,
                                                                                path=path,
                                                                                splits=splits)



    plot_profits_eval(running_profits=eval_running_profits,
                      total_profits=eval_total_profits,
                      mkt_total_profits=mkt_total_profits,
                      mkt_running_profits=mkt_running_profits,
                      path=plots_path)

    weights = f'weights/{dirname}/{index}/{dirname}_{symbol}.pt'
    experiment_log = {
        'index': index,
        'train': {
            'losses': losses,
            'rewards': rewards,
            'val rewards': val_rewards,
            'profits': profits,
            'val profits': val_profits
        },
        'eval': {
            'rewards': eval_rewards,
            'profits': eval_profits,
            'running profits': eval_running_profits,
            'total profits': eval_total_profits
        },
        'weights': weights
    }

    save_weights(model=model, OUT_PATH=weights)

    print(f"Logging Experiment for Numq Transfer Learning on index: {index} ... ")
    log_experiment(experiment=experiment_log, filename=f"{dirname}.nenq.{index}")


def plot_group_eval_results(results, path):
    for group in results:
        running_profits = results[group]["running profits"]
        plt.plot(list(range(len(running_profits))), running_profits, label=group)
    plt.title("Eval Profits")
    plt.legend()
    plt.savefig(f"plots/{path}/evaluation_all_groups.png")
    plt.close()

def plot_profits_eval(running_profits, total_profits, mkt_total_profits, mkt_running_profits, path):
    print(f"TOTAL MKT PROFITS : {mkt_total_profits}")
    print(f"TOTAL AGENT PROFITS : {total_profits}")
    plt.plot(list(range(len(running_profits))), running_profits, label="Trading Agent", color="blue")
    plt.plot(list(range(len(mkt_running_profits))), mkt_running_profits, label="MKT", color="black")
    plt.title("Eval Profits")
    plt.legend()
    plt.savefig(f"plots/{path}/evaluation.png")
    plt.close()

def plot_losses(losses, path):
    plt.plot(list(range(len(losses))), losses)
    plt.title("Losses")
    plt.savefig(f"plots/{path}/losses.png")
    plt.close()

def plot_rewards(rewards, val_rewards, path):
    plt.plot(list(range(len(rewards))), rewards, label="Training", color="lightblue")
    plt.plot(list(range(len(val_rewards))), val_rewards, label="Validation", color="blue")
    plt.title("Rewards")
    plt.legend()
    plt.savefig(f"plots/{path}/rewards.png")
    plt.close()

def plot_profits_train(profits, val_profits, mkt_train_total_profits, mkt_valid_total_profits, path):
    plt.plot(list(range(len(profits))), profits, label="Training", color="lightblue")
    plt.plot(list(range(len(val_profits))), val_profits, label="Validation", color="blue")
    plt.plot(list(range(len(val_profits))), len(val_profits) * [mkt_train_total_profits], label="MKT-Train",
             color="gray")
    plt.plot(list(range(len(val_profits))), len(val_profits) * [mkt_valid_total_profits], label="MKT-Valid",
             color="black")
    plt.title("Total Profits")
    plt.legend()
    plt.savefig(f"plots/{path}/profits.png")
    plt.close()
