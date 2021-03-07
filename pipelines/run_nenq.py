from pipelines.transfer_learning import gather_groups
from pipelines.run_numq import train, evaluate
from models.models import *
from typing import List, Dict
import yaml
import json
import os
import matplotlib.pyplot as plt

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

def load_weights(model: DQN, IN_PATH):
    model.policy_net.load_state_dict(torch.load(IN_PATH))
    model.hard_update()
    return model

def save_weights(model: DQN, OUT_PATH):
    torch.save(model.target_net.state_dict(), OUT_PATH)
    return

def pretrain_on_group(groups: Dict, index:str, method: str, group:str):

    experiment_log = {
        'name': f"{index}.{method}.{group}",
        'index': index,
        'method': method,
        'group': group,
        'total profits': 0
    }
    model = DQN(NUMQ)
    for i, symbol in enumerate(groups[index][method][group]):
        if i > 0:
            previous_weights = f'weights/numq/{index}/{method}/{group}/numq_{i}.pt'
            print(f"Loading weights for {group}, index: {index}, method: {method}, symbol: {symbol} ... ")
            model = load_weights(model=model, IN_PATH=previous_weights)

        # print(f"Start Train on group: {group}, index: {index}, method: {method}, symbol: {symbol} ... ")
        model, losses, rewards, val_rewards, profits, val_profits = train(model=model,
                                                                          index=index,
                                                                          symbol=symbol,
                                                                          episodes=10,
                                                                          dataset='train',
                                                                          path=config['STONK_PATH'],
                                                                          splits=config['STONK_INDEX_SPLITS'])

        print(f"Start Eval on group: {group}, index: {index}, method: {method}, symbol: {symbol} ... ")

        eval_rewards, eval_profits, eval_running_profits, eval_total_profits = evaluate(model=model,
                                                                                       index=index,
                                                                                       symbol=symbol,
                                                                                       dataset='valid',
                                                                                       path=config['STONK_PATH'],
                                                                                       splits=config['STONK_INDEX_SPLITS'])

        current_weights = f'weights/numq/{index}/{method}/{group}/numq_{i+1}.pt'
        experiment_log[symbol] = {
            'train': {
                'losses': losses,
                'rewards':rewards,
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
            'weights': current_weights
        }
        experiment_log["total profits"] += eval_total_profits
        # plots_path = f'plots/numq/{index}/{method}/{group}/numq_{symbol}_{i+1}.pt'

        # print(f"Saving weights for {group}, index: {index}, method: {method}, symbol: {symbol} ... ")
        save_weights(model=model, OUT_PATH=current_weights)

    pretrained_weights = f'weights/numq/{index}/{method}/{group}/numq_pretrain.pt'

    save_weights(model=model, OUT_PATH=pretrained_weights)
    experiment_log["pretrained weights"] = pretrained_weights

    print(f"Logging Experiment for group: {group}, index: {index}, method: {method} ... ")

    log_experiment(experiment=experiment_log, filename=f"{index}.{method}.{group}")

    return experiment_log

def log_experiment(experiment:Dict, filename:str, path:str=config["EXPERIMENT_LOGS_PATH"]):
    outpath = os.path.join(path, f"{filename}.json")
    with open(outpath, 'w') as outfile:
        json.dump(experiment, outfile, indent=4)

def evaluate_groups(groups: Dict, index:str):
    best_group = {
        'method': '',
        'group': '',
        'profit': float('-inf'),
        'pretrained weights': ''
    }
    for method in groups[index]:

        for group in groups[index][method]:

            print(f"Starting Experiment for group: {group}, index: {index}, method: {method} ... ")
            log = pretrain_on_group(groups=groups, index=index, method=method, group=group)

            if log["total profits"] > best_group["profit"]:
                best_group["method"] = method
                best_group["group"] = group
                best_group["profit"] = log["total profits"]
                best_group["pretrained weights"] = log["pretrained weights"]

    return best_group



def run_nenq_on_index(index, symbol, train_set='train', eval_set='valid',
                      path=config["STONK_PATH"], splits=config["STONK_INDEX_SPLITS"]):
    groups = gather_groups()
    best_group = evaluate_groups(groups=groups, index=index)
    previous_weights = best_group["pretrained weights"]

    model = DQN(NUMQ)
    model = load_weights(model=model, IN_PATH=previous_weights)

    print(f"Start Train on {index}, symbol: {symbol} ... ")
    model, losses, rewards, val_rewards, profits, val_profits = train(model=model,
                                                                      index=index,
                                                                      symbol=symbol,
                                                                      episodes=10,
                                                                      dataset=train_set,
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

    plots_path= f"numq/{index}"
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

    weights = f'weights/numq/{index}/numq_{symbol}.pt'
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

    log_experiment(experiment=experiment_log, filename=f"numq_transfer_learning_{index}")


def plot_profits_eval(running_profits, total_profits, mkt_total_profits, mkt_running_profits, path):
    print(f"TOTAL MKT PROFITS : {mkt_total_profits}")
    print(f"TOTAL AGENT PROFITS : {total_profits}")
    plt.plot(list(range(len(running_profits))), running_profits, label="Trading Agent", color="blue")
    plt.plot(list(range(len(mkt_running_profits))), mkt_running_profits, label="MKT", color="black")
    plt.title("Eval Profits")
    plt.legend()
    plt.savefig(f"plots/{path}/evaluation.png")

def plot_losses(losses, path):
    plt.plot(list(range(len(losses))), losses)
    plt.title("Losses")
    plt.savefig(f"plots/{path}/losses.png")

def plot_rewards(rewards, val_rewards, path):
    plt.plot(list(range(len(rewards))), rewards, label="Training", color="lightblue")
    plt.plot(list(range(len(val_rewards))), val_rewards, label="Validation", color="blue")
    plt.title("Rewards")
    plt.legend()
    plt.savefig(f"plots/{path}/rewards.png")

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

if __name__=='__main__':
    run_nenq_on_index(index='gspc', symbol='^GSPC', train_set='train', eval_set='valid',
                      path=config["STONK_PATH"], splits=config["STONK_INDEX_SPLITS"])

    run_nenq_on_index(index='djia', symbol='^DJI', train_set='train', eval_set='valid',
                      path=config["STONK_PATH"], splits=config["STONK_INDEX_SPLITS"])

    run_nenq_on_index(index='nasdaq', symbol='^IXIC', train_set='train', eval_set='valid',
                      path=config["STONK_PATH"], splits=config["STONK_INDEX_SPLITS"])

    run_nenq_on_index(index='nyse', symbol='^NYA', train_set='train', eval_set='valid',
                      path=config["STONK_PATH"], splits=config["STONK_INDEX_SPLITS"])