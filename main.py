from models.models import *
from pipelines.run_dqn import train, evaluate
from pipelines.run_dqn_nen import run_dqn_nen_on_index
import matplotlib.pyplot as plt
import yaml

with open("config.yml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


def load_weights(model: DQN, IN_PATH):
    model.policy_net.load_state_dict(torch.load(IN_PATH))
    model.hard_update()
    return model


def save_weights(model: DQN, OUT_PATH):
    torch.save(model.target_net.state_dict(), OUT_PATH)
    return


def run_evaluations(model: DQN, index: str, symbol: str, dataset: str, path:str, splits):
    rewards, profits, running_profits, total_profits = evaluate(model,
                                                                index=index,
                                                                symbol=symbol,
                                                                dataset=dataset,
                                                                path=path,
                                                                splits=splits)

    # MKT buying and holding 1 share
    mkt_rewards, mkt_profits, mkt_running_profits, mkt_total_profits = evaluate(model,
                                                                                index=index,
                                                                                symbol=symbol,
                                                                                dataset=dataset,
                                                                                strategy=0,
                                                                                strategy_num=1.0,
                                                                                use_strategy=True,
                                                                                only_use_strategy=True,
                                                                                path=path,
                                                                                splits=splits)

    print(f"TOTAL MKT PROFITS : {mkt_total_profits}")
    print(f"TOTAL AGENT PROFITS : {total_profits}")
    plt.plot(list(range(len(running_profits))), running_profits, label="Trading Agent", color="blue")
    plt.plot(list(range(len(mkt_running_profits))), mkt_running_profits, label="MKT", color="black")
    plt.legend()
    # plt.savefig("plots/evaluation.png")
    plt.savefig(f"plots/{index}_ep_{config['EPISODES']}_evaluation.png")
    plt.title("Eval Profits")
    plt.show()


def run_training(model: DQN, index: str, symbol: str, train_dataset: str, valid_dataset: str,
                 strategy: int, path:str, splits):
    model, losses, rewards, val_rewards, profits, val_profits = train(model=model,
                                                                      index=index,
                                                                      symbol=symbol,
                                                                      dataset=train_dataset,
                                                                      strategy=strategy,
                                                                      path=path,
                                                                      splits=splits)

    # MKT on training set
    print('MKT BUY on Train Set')
    mkt_train_rewards, mkt_train_profits, mkt_train_running_profits, mkt_train_total_profits = evaluate(model,
                                                                                                        index=index,
                                                                                                        symbol=symbol,
                                                                                                        dataset=train_dataset,
                                                                                                        strategy=0,
                                                                                                        strategy_num=1.0,
                                                                                                        only_use_strategy=True,
                                                                                                        path=path,
                                                                                                        splits=splits)

    # MKT on eval set
    print('MKT BUY on Eval Set')
    mkt_valid_rewards, mkt_valid_profits, mkt_valid_running_profits, mkt_valid_total_profits = evaluate(model,
                                                                                                        index=index,
                                                                                                        symbol=symbol,
                                                                                                        dataset=valid_dataset,
                                                                                                        strategy=0,
                                                                                                        strategy_num=1.0,
                                                                                                        only_use_strategy=True,
                                                                                                        path=path,
                                                                                                        splits=splits)

    plt.plot(list(range(len(losses))), losses)
    plt.title("Losses")
    plt.savefig(f"plots/{index}_ep_{config['EPISODES']}_losses.png")
    # plt.savefig("plots/losses.png")
    plt.show()

    plt.plot(list(range(len(rewards))), rewards, label="Training", color="lightblue")
    plt.plot(list(range(len(val_rewards))), val_rewards, label="Validation", color="blue")
    plt.title("Rewards")
    plt.savefig(f"plots/{index}_ep_{config['EPISODES']}_rewards.png")
    # plt.savefig("plots/rewards.png")
    plt.legend()
    plt.show()

    plt.plot(list(range(len(profits))), profits, label="Training", color="lightblue")
    plt.plot(list(range(len(val_profits))), val_profits, label="Validation", color="blue")
    plt.plot(list(range(len(val_profits))), len(val_profits) * [mkt_train_total_profits], label="MKT-Train", color="gray")
    plt.plot(list(range(len(val_profits))), len(val_profits) * [mkt_valid_total_profits], label="MKT-Valid", color="black")
    plt.title("Total Profits")
    plt.savefig(f"plots/{index}_ep_{config['EPISODES']}_profits.png")
    plt.legend()
    plt.show()
    return model


def run_experiment(**experiment):

    # GET TASK AND METHOD
    task = experiment.get('task', 'train')
    method = experiment['model'].get('method', NUMQ)

    # DATA
    index = experiment['data'].get('index', 'gspc')
    symbol = experiment['data'].get('symbol', '^GSPC')
    train_set = experiment['data'].get('train_set', 'train')
    eval_set = experiment['data'].get('eval_set', 'valid')
    path = experiment['data'].get('path', config['STONK_PATH'])
    splits = experiment['data'].get('splits', config['STONK_INDEX_SPLITS'])

    if task == 'transfer_learning':
        run_dqn_nen_on_index(model_method=method, index=index, symbol=symbol, train_set=train_set,
                             eval_set=eval_set, path=path, splits=splits)

    else:
        # create model
        model = DQN(method=method)

        # do we load weights before training or evaluating
        load = experiment['weights'].get('load_weights', False)
        if load:
            weights_in_path = experiment['weights'].get('weights_in_path', 'numq_test.pt')
            model = load_weights(model=model, IN_PATH=f'weights/{weights_in_path}')

        if task == 'evaluate':
            run_evaluations(model=model, index=index, symbol=symbol, dataset=eval_set, path=path, splits=splits)

        elif task =='train':
            # get strategy values
            strategy = experiment['model'].get('strategy', 1)

            model = run_training(model=model, index=index, symbol=symbol, train_dataset=train_set, valid_dataset=eval_set,
                                 strategy=strategy, path=path, splits=splits)

            run_evaluations(model=model, index=index, symbol=symbol, dataset=eval_set, path=path, splits=splits)

            # do we save weights after training
            save = experiment['weights'].get('save_weights', False)
            if save:
                weights_out_path = experiment['weights'].get('weights_out_path', 'numq_test.pt')
                save_weights(model=model, OUT_PATH=f'weights/{weights_out_path}')

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
        'weights': {
            'load_weights': False,
            'save_weights': False,
            # weights/{your_weights_here}
            'weights_in_path': 'numq_test.pt',
            'weights_out_path': 'numdreg_id_gspc.pt',
        }
    }

    run_experiment(**experiment)