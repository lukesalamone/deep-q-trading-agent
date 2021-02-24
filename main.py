# from pipelines.build_batches import batched
from models.models import *
# from pipelines.train import train_numq
from pipelines.train_dqn import train
from torch import tensor

print("Preparing data...")
# dataloader = batched('sx5e')

if __name__ == '__main__':
    lr = 1e-4
    model = DQN(NumQModel())
    threshold = 0.01
    gamma = 0.85
    # strategy = tensor(0)
    train(model=model, dataset='gspc', num_episodes=2, memory_capacity=3000, strategy=0)
    # dataloader = batched(dataset='gspc', batch_size=64)

    # train_numq(model, dataloader, threshold, gamma, lr, strategy)
