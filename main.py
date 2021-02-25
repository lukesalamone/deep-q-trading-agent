from models.models import *
from pipelines.run_dqn import train, evaluate
from torch import tensor
import matplotlib.pyplot as plt
import math
import os
import yaml

if __name__ == '__main__':
    # Get all config values and hyperparameters
    with open("config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile)

    model = DQN(NumQModel())
    model, losses = train(model, num_episodes=10, dataset='gspc')
    plt.plot(list(range(len(losses))), list(map(math.log, losses)))
    plt.title("Log Losses")
    plt.show()

    PATH = 'weights/numq_gspc_100.pt'
    torch.save(model.target_net.state_dict(), PATH)
    model = DQN(NumQModel())
    model.policy_net.load_state_dict(torch.load(PATH))
    model.transfer_weights()

    model, losses = train(model, num_episodes=10, dataset='gspc')
    plt.plot(list(range(len(losses))), list(map(math.log, losses)))
    plt.title("Log Losses")
    plt.show()

    torch.save(model.target_net.state_dict(), PATH)

    model = DQN(NumQModel())
    model.policy_net.load_state_dict(torch.load(PATH))
    model.transfer_weights()

    profits, total_profits = evaluate(model, dataset='gspc', evaluation_set='valid', strategy=config["STRATEGY"], only_use_strategy=False)

    print(f"TOTAL PROFITS : {total_profits}")
    plt.plot(list(range(len(profits))), profits)
    plt.title("Profits")
    plt.show()
