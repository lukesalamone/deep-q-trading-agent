from models.models import *
from pipelines.train_dqn import train
from torch import tensor

if __name__ == '__main__':
    model = DQN(NumQModel())
    train(model, num_episodes=5, dataset='gspc')
