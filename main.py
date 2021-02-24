from models.models import *
from pipelines.train_dqn import train
from torch import tensor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = DQN(NumQModel())
    losses = train(model, num_episodes=10, dataset='gspc')
    plt.plot(list(range(len(losses))), losses)
    plt.show()
