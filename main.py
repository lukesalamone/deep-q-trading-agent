from models.models import *
from pipelines.run_dqn import train, evaluate
from torch import tensor
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    model = DQN(NumQModel())
    losses = train(model, num_episodes=20, dataset='gspc')
    plt.plot(list(range(len(losses))), list(map(math.log, losses)))
    plt.title("Log Losses")
    plt.show()
