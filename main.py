from models.models import *
from pipelines.run_dqn import train, evaluate
from pipelines.run_experiments import *
from pipelines.run_dqn_nen import run_dqn_nen_on_index
import matplotlib.pyplot as plt
import yaml

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)

if __name__ == '__main__':
    # Input your experiment params
    experiment = {
        # train, evaluate, transfer_learning
        'task': 'transfer_learning',
        'data': {
            'index': 'djia',
            'symbol': '^DJI',
            'train_set': 'train',
            'eval_set': 'valid',
            'path': config['STONK_PATH'],
            'splits': config['STONK_INDEX_SPLITS']
        },
        'model': {
            # NUMQ, NUMDREG_ID, NUMDREG_AD
            'method': NUMDREG_ID,
            'strategy': 1
        },
        'training': {
            'episodes':config['EPISODES']
        },
        'weights': {
            'load_weights': False,
            'save_weights': False,
            # weights/{your_weights_here}
            'weights_in_path': 'numq_test.pt',
            'weights_out_path': 'numdreg_id_gspc.pt',
        }
    }

    run_experiment(**experiment)