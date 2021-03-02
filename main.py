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


if __name__ == '__main__':
    # Get all config values and hyperparameters
    with open("config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile)
    
    model = DQN(method=NUMQ)
    model, losses, rewards = train(model, num_episodes=10, dataset='gspc')

    plt.plot(list(range(len(losses))), losses)
    plt.title("Log Losses")
    plt.show()

    plt.plot(list(range(len(rewards))), rewards)
    plt.title("Rewards")
    plt.show()

    """
    PATH = 'weights/not_numq_gspc_20.pt'
    torch.save(model.target_net.state_dict(), PATH)

    model = DQN(NUMQ)
    model.policy_net.load_state_dict(torch.load(PATH))
    model.transfer_weights()
    """
    

    profits, running_profits, total_profits = evaluate(model, dataset='gspc', evaluation_set='valid', strategy=config["STRATEGY"], only_use_strategy=False)
    hold_profits, hold_running_profits, hold_total_profits = evaluate(model, dataset='gspc', evaluation_set='valid', strategy=config["STRATEGY"], only_use_strategy=True)

    print(f"TOTAL PROFITS : {total_profits}")
    plt.plot(list(range(len(running_profits))), running_profits, label="Model strategy")
    plt.plot(list(range(len(hold_running_profits))), hold_running_profits, label="Buy and hold")
    plt.legend()
    plt.title("Profits")
    plt.show()
