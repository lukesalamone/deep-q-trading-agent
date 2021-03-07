from pipelines.transfer_learning import gather_groups
from pipelines.run_numq import train, evaluate
from models.models import *
from typing import List, Dict
import yaml
import json
import os

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
    group_total_profits = []
    experiment_log = {
        'name': f"{index}.{method}.{group}",
        'index': index,
        'method': method,
        'group': group,
        'symbol': {},
        'total profits': 0
    }

    for i, symbol in enumerate(groups[index][method][group]):
        model = DQN(NUMQ)
        if i > 0:
            previous_weights = f'weights/numq/{index}/{method}/{group}/numq_{symbol}_{i}.pt'
            model = load_weights(model=model, IN_PATH=previous_weights)

        print(f"Start Train on group: {group}, index: {index}, method: {method}, symbol: {symbol} ... ")
        model, losses, rewards, val_rewards, profits, val_profits = train(model=model,
                                                                          index=index,
                                                                          symbol=symbol,
                                                                          episodes=10,
                                                                          dataset='train',
                                                                          path=config['STONK_PATH'],
                                                                          splits=config['STONK_INDEX_SPLITS'])

        print(f"Start Eval on group: {group}, index: {index}, method: {method}, symbol: {symbol} ... ")

        eval_rewards, eval_profits, eval_running_profits, eval_total_profit = evaluate(model=model,
                                                                                       index=index,
                                                                                       symbol=symbol,
                                                                                       dataset='valid',
                                                                                       path=config['STONK_PATH'],
                                                                                       splits=config['STONK_INDEX_SPLITS'])

        group_total_profits.append(eval_total_profit)
        current_weights = f'weights/numq/{index}/{method}/{group}/numq_{symbol}_{i+1}.pt'

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
                'total profits': eval_total_profit
            }
        }
        experiment_log["total profits"] += eval_total_profit

        # plots_path = f'plots/numq/{index}/{method}/{group}/numq_{symbol}_{i+1}.pt'

        save_weights(model=model, OUT_PATH=current_weights)

    print(f"Logging Experiment for group: {group}, index: {index}, method: {method} ... ")

    log_experiment(experiment=experiment_log, filename=f"{index}.{method}.{group}")

    return experiment_log
def log_experiment(experiment:Dict, filename:str, path:str=config["EXPERIMENT_LOGS_PATH"]):
    outpath = os.path.join(path, f"{filename}.json")
    with open(outpath, 'w') as outfile:
        json.dump(experiment, outfile)

def evaluate_groups(groups: Dict):
    best_group = ''
    max_profit = -9999999.9
    for index in groups:
        for method in groups[index]:
            for group in groups[index][method]:
                print(f"Starting Experiment for group: {group}, index: {index}, method: {method} ... ")
                log = pretrain_on_group(groups=groups, index=index, method=method, group=group)
                name, profit = log["name"], log["total profits"]
                if profit > max_profit:
                    best_group, max_profit = name, profit

    return best_group, max_profit

def main():
    groups = gather_groups()
    best_group, max_profit = evaluate_groups(groups=groups)

if __name__=='__main__':
    groups = gather_groups()
    print(groups)