from models.models import *
from pipelines.run_dqn import train, evaluate
from pipelines.run_experiments import *
from pipelines.run_dqn_nen import run_dqn_nen_on_index
import matplotlib.pyplot as plt
import yaml
import argparse

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

parser = argparse.ArgumentParser(description="Run the deep Q trading agent")
parser.add_argument('--task', type=str, help='The task we want to run (train, evaluate, transfer_learning)')
parser.add_argument('--epsd', type=int, help='The number of episodes to run')
parser.add_argument('--indx', type=str, help='The index we are trading (gspc, djia, nasdaq, nyse)')
parser.add_argument('--symb', type=str, help='The symbol of what we are trading')
parser.add_argument('--eval', type=str, help='The data set to evaluate on (train, valid, test)')
parser.add_argument('--mthd', type=str, help='The model method to use (numq, numdregad, numdregid)')
parser.add_argument('--stgy', type=str, help='The default strategy to use (buy, hold, sell)')
parser.add_argument('--load', type=str, help='Path to load model from')
parser.add_argument('--save', type=str, help='Path to save model')

args = parser.parse_args()

if __name__ == '__main__':
    # Helper dicts to set values
    method_dict = {'numq': NUMQ, 'numdregad':NUMDREG_AD, 'numdregid': NUMDREG_ID}
    strategy_dict = {'buy': 0, 'hold': 1, 'sell': 2}

    # Set default values
    task = 'evaluate'
    episodes = config['EPISODES']
    index = 'gspc'
    symbol = '^GSPC'
    eval_set = 'valid'
    method = method_dict['numq']
    strategy = strategy_dict['hold']
    load_path = 'temp_agent.pt'
    save_path = 'temp_agent.pt'

    # Get args if specified
    if args.task:
        task = args.task
    if args.epsd:
        episodes = args.epsd
    if args.indx:
        index = args.indx
    if args.symb:
        symbol = args.symb
    if args.eval:
        eval_set = args.eval
    if args.mthd:
        method = method_dict[args.mthd]
    if args.stgy:
        strategy = strategy_dict[args.stgy]
    if args.load:
        load_path = args.load
    if args.save:
        save_path = args.save


    # Input your experiment params
    experiment = {
        # train, evaluate, transfer_learning
        'task': task,
        'data': {
            'index': index,
            'symbol': symbol,
            'train_set': 'train',
            'eval_set': eval_set,
            'path': config['STONK_PATH'],
            'splits': config['STONK_INDEX_SPLITS']
        },
        'model': {
            # NUMQ, NUMDREG_ID, NUMDREG_AD
            'method': method,
            'strategy': strategy
        },
        'training': {
            'episodes':episodes
        },
        'weights': {
            'load_weights': args.load,
            'save_weights': args.save,
            # weights/{your_weights_here}
            'weights_in_path': load_path,
            'weights_out_path': save_path,
        }
    }

    run_experiment(**experiment)