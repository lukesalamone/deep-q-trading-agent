from models.models import *
from pipelines.run_dqn import train, evaluate
from torch import tensor
import matplotlib.pyplot as plt
import math
import os
import yaml

def run_experiment():
    # creates model
    #
    pass

def load_weights(model:DQN, IN_PATH):
    model.policy_net.load_state_dict(torch.load(PATH))
    model.transfer_weights()
    return model

def save_weights(model:DQN, OUT_PATH):
    torch.save(model.target_net.state_dict(), OUT_PATH)
    return

if __name__ == '__main__':
    # Get all config values and hyperparameters
    with open("config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile)

    PATH = 'weights/numq_gspc_100.pt'
    model = DQN(method=NUMQ)

    model.policy_net.load_state_dict(torch.load(PATH))
    model.transfer_weights()

    model, losses, rewards, profits = train(model, dataset='gspc')

    plt.plot(list(range(len(losses))), losses)
    plt.title("Losses")
    plt.show()

    plt.plot(list(range(len(rewards))), rewards)
    plt.title("Rewards")
    plt.show()

    plt.plot(list(range(len(profits))), profits)
    plt.title("Total Profits")
    plt.show()

    PATH = 'weights/not_numq_gspc_20.pt'
    torch.save(model.target_net.state_dict(), PATH)

    """
    model = DQN(NUMQ)
    model.policy_net.load_state_dict(torch.load(PATH))
    model.transfer_weights()
    """

    profits, running_profits, total_profits = evaluate(model, dataset='gspc', evaluation_set='test', strategy=config["STRATEGY"], only_use_strategy=False)
    hold_profits, hold_running_profits, hold_total_profits = evaluate(model, dataset='gspc', evaluation_set='test', strategy=config["STRATEGY"], only_use_strategy=True)

    print(f"TOTAL PROFITS : {total_profits}")
    plt.plot(list(range(len(running_profits))), running_profits, label="Model strategy")
    plt.plot(list(range(len(hold_running_profits))), hold_running_profits, label="Buy and hold")
    plt.legend()
    plt.title("Profits")
    plt.show()
